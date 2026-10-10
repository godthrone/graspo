"""flow.backend.native.models — native 模型族实现：按家族分子目录，common 承载共享件。

★ 本包是「模型 / LoRA 适配」（一名一物里的第三种），**不是** `flow/adapter/` 的对接契约。
每个模型族：``model_adapter.py``（适配器）+ ``model.py`` / ``layers*.py`` / ``generation*.py`` /
``training*.py``（模型实现）。
"""
