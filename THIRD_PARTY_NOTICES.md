# Third-Party Notices

GRASPO keeps the algorithm implementation in this repository. The production
backend is `megatron-native`: GRASPO owns rollout, retry/filtering,
ReplayBuffer, reward, advantage, loss, logging, and checkpoint behavior.

External dependencies are used through their public Python APIs:

- PyTorch: tensor computation, autograd, distributed process groups, and
  optimization primitives.
- Transformers: tokenizer/config loading and the small `hf-reference` backend.
- safetensors: safe model/checkpoint tensor loading and saving.
- PyYAML: configuration loading.
- pandas: optional local spreadsheet data preparation utility.
- PEFT: optional LoRA reference/export helper. The production
  `megatron-native` training path must not rely on Accelerate at runtime.
- Megatron-LM / Megatron Core: optional open-source tensor-parallel runtime
  dependency/reference for large-model training. Do not vendor Megatron source
  into this repository without preserving its license and notices.

The production training path does not depend on NVIDIA NeMo, NeMo-RL, NGC NeMo
containers, vLLM, Ray, DeepSpeed, FSDP, DDP, Accelerate, TransformerEngine, Apex,
or ZeRO-style fallbacks.

If GRASPO vendors any third-party source file in the future, keep that file's
copyright header and update this notice with the exact source and license.
