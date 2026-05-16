# GRASPO Documentation

GRASPO is a standalone, model-agnostic training project for structured-output
language models, primarily designed for industrial Agent workloads. It is not
intended to replace every general-purpose rollout RL algorithm. Instead, it
focuses on tasks with explicit structure and verifiable fields, such as
information extraction, classification, JSON generation, and tool calls.

The algorithm grew out of practical information-extraction work where labels
were provided by field engineers and operations experts. The data volume is
limited, but the samples are high quality and expensive to collect and review.
The core value of GRASPO is not generic RLHF, but making Agent structured outputs
more reliable with modest data and moderate GPU cost. ARD is added to reduce
general-capability forgetting after hard-sample SFT.

## Contents

- [Quickstart](quickstart.md)
- [Configuration](configuration.md)
- [Data Format](data-format.md)
- [Training](training.md)
- [Docker](docker.md)
- [Troubleshooting](troubleshooting.md)
- [Algorithm](algorithm.md)
- [Anchor Replay Distillation](ard.md)

## Scope

v0.1 targets Hugging Face `AutoModelForCausalLM` text models. Qwen, Llama,
DeepSeek, Mistral, and similar models are examples, not special cases in the core
algorithm.

## Suitable Tasks

- information extraction into JSON
- structured classification with verifiable labels
- Agent tool-call argument generation
- form, ticket, and log parsing
- any task that can be scored by rules, tests, or field-level metrics

GRASPO is not the first choice for open-ended writing, subjective preference
alignment, or tasks without field-level scoring. LLM-assisted field scoring is a
possible future extension, but v0.1 focuses on rule-verifiable tasks.
