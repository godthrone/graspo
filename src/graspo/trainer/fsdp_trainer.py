from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graspo.core.advantage import group_advantages, has_reward_variance
from graspo.core.buffer import Experience, ReplayBuffer
from graspo.core.data import load_jsonl
from graspo.core.reward import GraspoReward
from graspo.core.schema import GraspoConfig, Sample
from graspo.trainer.checkpoint import save_lora_adapter
from graspo.trainer.generation import ensure_tokenizer_ready, generate_group
from graspo.trainer.lora import build_peft_config
from graspo.trainer.loss import GRASPOLoss, sequences_log_probs


@dataclass(slots=True)
class TrainStats:
    total: int = 0
    perfect_skipped: int = 0
    trainable_perfect: int = 0
    trainable_imperfect: int = 0
    invalid: int = 0
    retries: int = 0


def collate_samples(items: list[Sample]) -> list[Sample]:
    return items


def collate_experiences(items: list[Experience]) -> Experience:
    from torch.nn.utils.rnn import pad_sequence

    def pad(values, padding_value=0):
        return pad_sequence(values, batch_first=True, padding_value=padding_value)

    return Experience(
        sequences=pad([item.sequences for item in items]),
        old_log_probs=pad([item.old_log_probs for item in items]),
        advantages=pad([item.advantages for item in items], padding_value=0.0),
        attention_mask=pad([item.attention_mask for item in items]).bool(),
        action_mask=pad([item.action_mask for item in items]).bool(),
        rewards=torch.stack([item.rewards for item in items]),
    )


class FSDPGraspoTrainer:
    def __init__(self, config: GraspoConfig) -> None:
        self.config = config
        self.reward = GraspoReward(config.reward)
        self.replay_buffer = ReplayBuffer()
        self.stats = TrainStats()

    def train(self) -> None:
        from accelerate import Accelerator
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        accelerator = Accelerator()
        set_seed(self.config.training.seed)
        random.seed(self.config.training.seed + accelerator.process_index)

        model_path = self.config.model.model_path
        if not model_path or model_path.startswith("${"):
            raise ValueError("Set model.model_path in config or MODEL_PATH environment variable")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        ensure_tokenizer_ready(tokenizer)

        torch_dtype = self._resolve_dtype(self.config.model.torch_dtype)
        model_kwargs = {
            "trust_remote_code": self.config.model.trust_remote_code,
            "torch_dtype": torch_dtype,
        }
        if self.config.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.model.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if self.config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False

        peft_config = build_peft_config(self.config, model)
        model = get_peft_model(model, peft_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

        samples = load_jsonl(self.config.data.train_path)
        dataloader = DataLoader(
            samples,
            batch_size=self.config.training.prompts_per_rank,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            collate_fn=collate_samples,
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
        loss_fn = GRASPOLoss(self.config.training.clip_eps)

        global_step = 0
        output_dir = Path(self.config.training.output_dir)
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.training.total_epochs):
            for batch in dataloader:
                for sample in batch:
                    self.stats.total += 1
                    self._sample_one(accelerator, model, tokenizer, sample)

                threshold = self.config.training.train_batch_size * self.config.training.group_size
                if len(self.replay_buffer) >= threshold:
                    global_step += 1
                    self._optimize(accelerator, model, optimizer, loss_fn)
                    self._log(accelerator, epoch, global_step)

                    if global_step % self.config.training.save_steps == 0:
                        self._save(accelerator, model, tokenizer, output_dir / f"step_{global_step}")

                    if 0 < self.config.training.max_steps <= global_step:
                        self._save(accelerator, model, tokenizer, output_dir / "final")
                        return

            if len(self.replay_buffer) > 0:
                global_step += 1
                self._optimize(accelerator, model, optimizer, loss_fn)
                self._log(accelerator, epoch, global_step)

        self._save(accelerator, model, tokenizer, output_dir / "final")

    def _sample_one(self, accelerator, model, tokenizer, sample: Sample) -> None:
        device = accelerator.device
        accepted = False
        best_payload = None
        dummy_prompt = " "
        rewards: list[float] = []

        model.eval()
        for attempt in range(self.config.training.max_retry + 1):
            active_prompt = sample.prompt if not accepted else dummy_prompt
            sequences, attention_mask, action_mask, completions, _ = generate_group(
                model=model,
                tokenizer=tokenizer,
                prompt=active_prompt,
                group_size=self.config.training.group_size,
                device=device,
                max_new_tokens=self.config.training.max_new_tokens,
                max_prompt_length=self.config.data.max_prompt_length,
                temperature=self.config.training.temperature,
                top_p=self.config.training.top_p,
                synced_gpus=accelerator.num_processes > 1,
                chat_template_kwargs=self.config.model.chat_template_kwargs,
            )
            if accepted:
                continue

            results = [self.reward.score(text, sample.ground_truth) for text in completions]
            rewards = [result.reward for result in results]
            median_reward = sorted(rewards)[len(rewards) // 2]
            max_reward = max(rewards)

            if attempt == 0 and median_reward >= self.config.training.perfect_reward_threshold:
                self.stats.perfect_skipped += 1
                accepted = True
                continue

            best_payload = (sequences, attention_mask, action_mask, rewards)
            if max_reward >= self.config.training.perfect_reward_threshold:
                accepted = True
                self.stats.trainable_perfect += 1
                continue
            self.stats.retries += 1

        if best_payload is None:
            return

        if not has_reward_variance(rewards):
            self.stats.invalid += 1
            return

        if max(rewards) < self.config.training.perfect_reward_threshold:
            self.stats.trainable_imperfect += 1

        sequences, attention_mask, action_mask, rewards = best_payload
        with torch.no_grad():
            old_log_probs = sequences_log_probs(model, sequences, attention_mask).detach()
        advantages = torch.tensor(
            group_advantages(rewards),
            dtype=old_log_probs.dtype,
            device=old_log_probs.device,
        ).unsqueeze(1)
        advantages = advantages.expand_as(old_log_probs)
        reward_tensor = torch.tensor(rewards, dtype=old_log_probs.dtype, device=old_log_probs.device)

        items: list[Experience] = []
        for idx in range(sequences.shape[0]):
            items.append(
                Experience(
                    sequences=sequences[idx].detach().cpu(),
                    old_log_probs=old_log_probs[idx].detach().cpu(),
                    advantages=advantages[idx].detach().cpu(),
                    attention_mask=attention_mask[idx].detach().cpu(),
                    action_mask=action_mask[idx].detach().cpu(),
                    rewards=reward_tensor[idx].detach().cpu(),
                )
            )
        self.replay_buffer.append_many(items)

    def _optimize(self, accelerator, model, optimizer, loss_fn: GRASPOLoss) -> None:
        local_count = torch.tensor(len(self.replay_buffer), device=accelerator.device)
        min_count = self._distributed_min(local_count, accelerator).item()
        usable = int(min_count)
        if usable < self.config.training.train_batch_size:
            self.replay_buffer.clear()
            return

        usable -= usable % self.config.training.train_batch_size
        data = self.replay_buffer.take(usable)
        loader = DataLoader(
            data,
            batch_size=self.config.training.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_experiences,
        )

        model.train()
        for _ in range(self.config.training.epochs_per_step):
            for exp in loader:
                exp = Experience(
                    sequences=exp.sequences.to(accelerator.device),
                    old_log_probs=exp.old_log_probs.to(accelerator.device),
                    advantages=exp.advantages.to(accelerator.device),
                    attention_mask=exp.attention_mask.to(accelerator.device),
                    action_mask=exp.action_mask.to(accelerator.device),
                    rewards=exp.rewards.to(accelerator.device),
                )
                optimizer.zero_grad(set_to_none=True)
                log_probs = sequences_log_probs(model, exp.sequences, exp.attention_mask)
                loss = loss_fn(log_probs, exp.old_log_probs, exp.advantages, exp.action_mask)
                if not torch.isfinite(loss):
                    continue
                accelerator.backward(loss)
                if self.config.training.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
                optimizer.step()
        self.replay_buffer.clear()

    def _log(self, accelerator, epoch: int, step: int) -> None:
        if not accelerator.is_main_process:
            return
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "step": step,
                    "stats": self.stats.__dict__,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def _save(self, accelerator, model, tokenizer, path: Path) -> None:
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            save_lora_adapter(unwrapped, tokenizer, path)
        accelerator.wait_for_everyone()

    @staticmethod
    def _resolve_dtype(name: str):
        lowered = name.lower()
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp16", "float16"}:
            return torch.float16
        if lowered in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {name}")

    @staticmethod
    def _distributed_min(value: torch.Tensor, accelerator) -> torch.Tensor:
        if accelerator.num_processes <= 1:
            return value
        import torch.distributed as dist

        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return value
