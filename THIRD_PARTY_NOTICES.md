# Third-Party Notices

GRASPO keeps the training algorithm implementation in this repository. The
production backend is `native-tp`: GRASPO owns rollout, retry/filtering,
ReplayBuffer, reward, advantage, loss, logging, LoRA, checkpoint behavior, and
uses PyTorch distributed process groups for tensor parallelism.

External dependencies are used through their public Python APIs:

- PyTorch: tensor computation, autograd, optimizer primitives, CUDA/NCCL access,
  and distributed process groups.
- Transformers: tokenizer/config loading and the small `hf-reference` backend.
- safetensors: safe model/checkpoint tensor loading and saving.
- PyYAML: configuration loading.
- pandas and datasets: optional local data preparation utilities.
- PEFT: optional LoRA reference/export/import helper. The production `native-tp`
  training path must not wrap the model with PEFT or rely on Accelerate at
  runtime.

The production training path does not depend on Megatron, NVIDIA NeMo, NeMo-RL,
NGC NeMo containers, vLLM, Ray, DeepSpeed, FSDP, DDP, Accelerate,
TransformerEngine, Apex, or ZeRO-style fallbacks.

If GRASPO vendors any third-party source file in the future, keep that file's
copyright header and update this notice with the exact source and license.