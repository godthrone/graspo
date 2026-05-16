# GRASPO 中文文档

GRASPO 是一个独立、模型无关的结构化输出强化学习训练项目，主要面向工业界 Agent 场景。它不是试图替代所有通用 rollout 强化学习算法，而是专注于信息提取、分类、JSON 生成、tool call 等“输出结构明确、字段可以验证”的任务。

这个算法最初来自信息提取实践：业务标注数据来自现场工程师和运维专家，数量有限但质量高，采集和复核成本都很高。GRASPO 的核心价值不在于追求通用 RLHF，而在于用较少数据、较低显存成本，把 Agent 的结构化输出行为训练得更可靠。ARD 则用于缓解困难样本 SFT 之后的通用能力遗忘。

## 目录

- [快速开始](quickstart.md)
- [配置说明](configuration.md)
- [数据格式](data-format.md)
- [训练](training.md)
- [Docker](docker.md)
- [排障](troubleshooting.md)
- [算法说明](algorithm.md)
- [Anchor Replay Distillation](ard.md)

## 支持范围

v0.1 面向 Hugging Face `AutoModelForCausalLM` 文本模型。Qwen、Llama、DeepSeek、Mistral 等只是示例模型族，核心算法不绑定任何具体模型。

## 适用任务

- 信息提取：从文本中抽取字段并输出 JSON
- 结构化分类：输出可验证的类别或标签
- Agent tool call：训练模型稳定生成工具调用参数
- 表单/工单/日志解析：要求字段准确、格式稳定、少废话
- 其它可通过规则、测试或字段级打分验证的任务

对于开放式写作、主观偏好对齐、无法定义字段级得分的任务，GRASPO 不是首选。未来可以接入大模型辅助评分作为字段级 reward，但 v0.1 先聚焦规则可验证任务。
