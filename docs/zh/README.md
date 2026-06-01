# GRASPO 中文文档

GRASPO 是一个独立的结构化输出训练项目，面向 JSON 抽取、分类、表单解析、
tool call 参数生成等字段可验证任务。

## 目录

- [快速开始](quickstart.md)
- [配置说明](configuration.md)
- [工程实现说明](engineering-implementation.md)
- [数据格式](data-format.md)
- [训练](training.md)
- [Docker](docker.md)
- [排障](troubleshooting.md)
- [算法说明](algorithm.md)

## 支持范围

v0.1 只有一条生产路线：`megatron-native`。它使用开源 Megatron-LM/Core
tensor parallel，GRASPO 自己控制算法流程。`hf-reference` 只用于单进程算法
对齐和小模型调试。

Qwen、Llama、DeepSeek、Mistral 等只是示例模型族，核心算法不绑定具体模型。
