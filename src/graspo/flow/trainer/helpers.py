"""GraspoFlowTrainer 纯函数工具（设施层）。

不依赖 self 状态，可独立测试。仅含设施层函数：时间戳、随机种子、配置备份、
tensor 提取、元数据格式化。
算法层函数（advantage 计算、统计序列化、group_stats、reward_detail）已迁入
ripple/ 对应模块。
"""

import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Any

# ── 时间戳 ────────────────────────────────────────────────────────────────────


def _timestamp() -> str:
    """返回当前时区的 ISO 格式时间戳。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ── 随机种子 ──────────────────────────────────────────────────────────────────


def _set_random_seed(seed: int, *, rank: int = 0) -> None:
    """设置所有随机数生成器的种子，确保可复现性。"""
    import numpy as np
    import torch

    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


# ── 配置备份 ──────────────────────────────────────────────────────────────────


def _backup_config(config: Any, output_dir: Any) -> None:
    """将当前配置写入输出目录，确保事后可完整复现。"""
    import yaml

    config_path = output_dir / "config.yaml"
    config_path.write_text(
        yaml.dump(config.model_dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# ── 生成数据提取 ──────────────────────────────────────────────────────────────


def generated_token_counts(generation: Any) -> list[int]:
    """从 generation 中提取每条 completion 的生成 token 数。

    当 action_mask 不可用时返回空列表——这是透明降级（不改变训练结果），不影响训练，
    仅影响监控日志中的 token 计数。降级原因通过 warnings 告知用户。
    """
    try:
        return [int(value) for value in generation.action_mask.detach().sum(dim=1).cpu().tolist()]
    except (AttributeError, TypeError, RuntimeError):
        logging.getLogger("graspo.trainer").warning(
            "generated_token_counts: action_mask unavailable, token counts will be empty"
        )
        return []


# ── 元数据处理 ─────────────────────────────────────────────────────────────────


def safe_sample_metadata(sample: Any) -> dict[str, Any]:
    """从 sample 中提取安全可输出的元数据（不含媒体原始数据）。"""
    redacted = {
        key: value
        for key, value in sample.metadata.items()
        if key not in {"image", "images", "video", "videos"}
    }
    if sample.media:
        counts: dict[str, int] = {}
        for item in sample.media:
            media_type = str(item.get("type") or "unknown")
            counts[media_type] = counts.get(media_type, 0) + 1
        redacted["media"] = {
            "count": len(sample.media),
            "types": counts,
        }
    return redacted


def public_generation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """从生成元数据中提取公开可输出的部分。"""
    public = {key: value for key, value in metadata.items() if not str(key).startswith("_")}
    private_rows = metadata.get("_multimodal_rows")
    if isinstance(private_rows, list) and private_rows:
        media_counts: dict[str, int] = {}
        for row in private_rows:
            if not isinstance(row, dict):
                continue
            for item in row.get("media") or []:
                if not isinstance(item, dict):
                    continue
                media_type = str(item.get("type") or "unknown")
                media_counts[media_type] = media_counts.get(media_type, 0) + 1
        public["multimodal"] = {
            "row_count": len(private_rows),
            "media_counts": media_counts,
        }
    return public


def experience_metadata_for_row(
    metadata: dict[str, Any] | None, row_index: int
) -> dict[str, Any] | None:
    """从生成元数据中提取第 row_index 条 experience 的元数据。"""
    if not metadata:
        return None
    rows = metadata.get("_multimodal_rows")
    if isinstance(rows, list):
        if row_index >= len(rows):
            raise RuntimeError(
                f"generation multimodal metadata has {len(rows)} rows, cannot index row {row_index}"
            )
        return {"_multimodal_rows": [rows[row_index]]}
    return dict(metadata)


# ── tool-call 辅助 ─────────────────────────────────────────────────────────────
# tool_call_count_mismatch_count / _target_tool_call_counts 已迁入
# ripple/parsing/classification.py（算法层，供 flow 与 ripple 共用）。


# ── 生成数据提取 ──────────────────────────────────────────────────────────────


def raw_generation_payload(generation: Any) -> dict[str, Any]:
    """提取 generation 的原始 tensor 数据（写入 raw 日志用）。"""
    return {
        "sequences": generation.sequences,
        "attention_mask": generation.attention_mask,
        "action_mask": generation.action_mask,
        "prompt_len": generation.prompt_len,
    }


# ── 输出目录准备（多 rank 安全，唯一入口）─────────────────────────────────────

#: 广播失败信息的定长字节数（够放 prepare_output_dir 的完整可操作报错）。
_ERROR_MESSAGE_BYTES = 4096


def prepare_output_dir_once(
    output_dir: Any,
    *,
    overwrite: bool,
    is_primary: bool,
) -> Any:
    """准备输出目录：**只有 primary rank 清目录，其余 rank 等它做完**（§1.4 唯一入口）。

    缺陷（228 实测 `T030` run2）：每个 rank 在 ``train()`` 开头各自调用
    ``prepare_output_dir(overwrite=True)`` ⇒ 4 个 rank **并发** ``shutil.rmtree``
    同一个目录，rank3 在 ``shutil._rmtree_safe_fd`` 里 ``os.lstat`` 到已被别的
    rank 删掉的目录 ⇒ ``FileNotFoundError: PosixPath('/out/T030')``。

    修法（**不是**"各 rank 各删一次"，那只是把竞态窗口缩窄）：
    - 只由 primary rank（``is_primary``，即全局 rank 0）执行破坏性清目录（§2.4）；
    - 其余 rank 在 ``dist.broadcast`` 上等待它完成——广播本身就是"清目录已完成"
      的同步点，因此没有任何 rank 会在 ``rmtree`` 窗口内建目录（阻断 B 的老问题
      也不会以多 rank 形态复发）；
    - 单卡或进程组未初始化时退化为直接调用 ``prepare_output_dir``，行为与旧实现
      逐字相同；
    - primary 的失败（含 ``overwrite=False`` 且目录非空）连同**原始报错文本**
      一起广播，全体 rank 抛同类异常 ⇒ fail-closed，且非主 rank **不会**在集合点
      上干等到 NCCL 看门狗超时（§2.3 边界校验即防呆）。

    语义不变：``overwrite=True`` 仍然清掉上次的全部产物；``overwrite=False`` 且
    目录非空仍然 ``FileExistsError``（预授权退路 §3.3）。
    """
    from graspo.flow.lora.lora_io import prepare_output_dir

    import torch
    import torch.distributed as dist

    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if not distributed:
        return prepare_output_dir(output_dir, overwrite=overwrite)

    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # 0=成功；1=目录非空且未授权覆盖（FileExistsError）；2=其他异常
    status = torch.zeros(1, dtype=torch.int32, device=device)
    message = torch.zeros(_ERROR_MESSAGE_BYTES, dtype=torch.uint8, device=device)
    if is_primary:
        try:
            prepare_output_dir(output_dir, overwrite=overwrite)
        except FileExistsError as exc:
            status.fill_(1)
            payload = str(exc).encode("utf-8")[:_ERROR_MESSAGE_BYTES]
            message[: len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(
                device
            )
        except Exception as exc:  # noqa: BLE001 —— 必须把失败带给所有 rank，避免其干等
            status.fill_(2)
            payload = f"{type(exc).__name__}: {exc}".encode("utf-8")[:_ERROR_MESSAGE_BYTES]
            message[: len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(
                device
            )
    dist.broadcast(status, src=0)
    dist.broadcast(message, src=0)
    code = int(status.item())
    if code == 1:
        raise FileExistsError(bytes(message.tolist()).decode("utf-8", "replace").rstrip("\x00"))
    if code == 2:
        raise RuntimeError(
            "rank0 准备输出目录失败：" + bytes(message.tolist()).decode("utf-8", "replace").rstrip("\x00")
        )
    return Path(output_dir)
