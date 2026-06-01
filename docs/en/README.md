# GRASPO Documentation

GRASPO is a standalone training project for structured-output language models.
It focuses on tasks with explicit structure and verifiable fields, such as JSON
extraction, classification, form parsing, and tool-call argument generation.

## Contents

- [Quickstart](quickstart.md)
- [Configuration](configuration.md)
- [Data Format](data-format.md)
- [Training](training.md)
- [Docker](docker.md)
- [Troubleshooting](troubleshooting.md)
- [Algorithm](algorithm.md)

## Scope

v0.1 has one production route: `megatron-native`, which uses open-source
Megatron-LM/Core tensor parallelism while GRASPO owns the algorithm control
flow. `hf-reference` is retained only for single-process algorithm parity and
small-model debugging.

Qwen, Llama, DeepSeek, Mistral, and similar model families are examples, not
special cases in the core algorithm.
