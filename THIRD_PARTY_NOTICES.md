# Third-Party Notices

GRASPO keeps the algorithm implementation in this repository. The recommended
large-model backend is `megatron-native`, where GRASPO owns rollout,
retry/filter/replay, reward, advantage, loss, and logging. Megatron Core/L.M. is
used only as an optional external tensor-parallel runtime dependency.

- Megatron-LM / Megatron Core is an optional external large-model tensor
  parallel dependency/reference. Do not vendor Megatron source into this
  repository without preserving its license and notices.
- NVIDIA NeMo, NeMo-RL, vLLM, Ray, DeepSpeed, FSDP, DDP, and Accelerate are
  not production training dependencies of this repository.
- flash-linear-attention and causal-conv1d may be used by specific model
  families for fast GDN/linear-attention execution.

If GRASPO vendors any third-party source file in the future, keep that file's
copyright header and update this notice with the exact source and license.
