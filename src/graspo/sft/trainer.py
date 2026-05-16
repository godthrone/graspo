from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graspo.sft.collator import ARDDataCollator
from graspo.sft.data import load_mixed_sft_samples
from graspo.sft.loss import anchor_kl_loss, weighted_ce_loss
from graspo.sft.schema import ARDSFTConfig
from graspo.trainer.checkpoint import save_lora_adapter
from graspo.trainer.generation import ensure_tokenizer_ready
from graspo.trainer.lora import build_peft_config


class ARDSFTTrainer:
    def __init__(self, config: ARDSFTConfig) -> None:
        self.config = config

    def train(self) -> None:
        from accelerate import Accelerator
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        accelerator = Accelerator()
        set_seed(self.config.training.seed)
        random.seed(self.config.training.seed + accelerator.process_index)

        model_path = self.config.model.model_path
        if not model_path or model_path.startswith("${"):
            raise ValueError("Set model.model_path or MODEL_PATH")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        ensure_tokenizer_ready(tokenizer)
        dtype = self._resolve_dtype(self.config.model.torch_dtype)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=dtype,
        )
        if self.config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
        peft_config = build_peft_config(self.config, model)
        model = get_peft_model(model, peft_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

        teacher = None
        if self.config.kl_distillation.enabled:
            teacher = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=self.config.model.trust_remote_code,
                torch_dtype=dtype,
            )
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)

        samples = load_mixed_sft_samples(
            self.config.data.hard_train_path,
            self.config.data.anchor_train_path,
        )
        collator = ARDDataCollator(
            tokenizer,
            max_length=self.config.data.max_length,
            chat_template_kwargs=self.config.model.chat_template_kwargs,
        )
        dataloader = DataLoader(
            samples,
            batch_size=self.config.training.per_device_batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            collate_fn=collator,
            drop_last=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        if teacher is not None:
            model, teacher, optimizer, dataloader = accelerator.prepare(model, teacher, optimizer, dataloader)
        else:
            model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

        output_dir = Path(self.config.training.output_dir)
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)

        global_step = 0
        for epoch in range(self.config.training.total_epochs):
            for batch in dataloader:
                global_step += 1
                batch = {key: value.to(accelerator.device) for key, value in batch.items()}
                sample_weights = torch.where(
                    batch["is_anchor"],
                    torch.full_like(batch["sample_weights"], self.config.training.anchor_ce_weight),
                    batch["sample_weights"],
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = weighted_ce_loss(outputs.logits, batch["labels"], sample_weights)

                if teacher is not None and self.config.kl_distillation.weight > 0:
                    with torch.no_grad():
                        teacher_outputs = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                        )
                    loss = loss + self.config.kl_distillation.weight * anchor_kl_loss(
                        outputs.logits,
                        teacher_outputs.logits,
                        batch["labels"],
                        batch["is_anchor"],
                        temperature=self.config.kl_distillation.temperature,
                    )

                accelerator.backward(loss)
                if self.config.training.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), self.config.training.max_grad_norm)
                optimizer.step()

                if accelerator.is_main_process and global_step % self.config.training.logging_steps == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "loss": float(loss.detach().cpu()),
                                "anchor_count": int(batch["is_anchor"].sum().detach().cpu()),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if global_step % self.config.training.save_steps == 0:
                    self._save(accelerator, model, tokenizer, output_dir / f"step_{global_step}")
                if 0 < self.config.training.max_steps <= global_step:
                    self._save(accelerator, model, tokenizer, output_dir / "final")
                    return

        self._save(accelerator, model, tokenizer, output_dir / "final")

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

