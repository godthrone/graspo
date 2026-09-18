"""训练 worker 进程入口：按 train_method + backend 分派 SFT/RL 训练器（支持 --smoke）。

**分派真相源**：``core.discovery.resolve_backend_builder``。
SFT 与 RL 各有一张注册表（``graspo.sft_backends`` / ``graspo.backends``），都在
``pyproject.toml`` 的 entry_points 里声明，开发模式下回退到 ``_DEV_FALLBACKS``。
因此新增后端（含"未来第三个后端"）不需要改动本文件的任何现有分支
（宪法 §1.2 对扩展开放、对修改关闭；决策 D5）。

**形状契约**：SFT 注册表里的每个工厂都满足
``factory(config, selection) -> 含 train(smoke: bool) 的训练器``。
native 侧由 ``create_native_sft_trainer`` 提供，msswift 侧由
``create_msswift_sft_trainer``（延迟构造器）提供。RL 侧沿用既有
``create_trainer`` 契约（同样返回含 ``train(smoke)`` 的训练器）。
"""

import argparse

from graspo.core.gpu_guard import require_gpu_lock_or_exit
from graspo.core.schema import GraspoConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m graspo.cli.train_worker",
        description="Internal GRASPO training worker. Use `graspo launch --config ...`.",
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke boundary (infrastructure param): run until the first optimize"
            " step then stop. Never mutates the config object."
        ),
    )
    args = parser.parse_args()

    # 锁卡守卫（fail-closed）：训练进程启动的第一件事。未显式锁卡 / 含生产卡
    # GPU6,7 / 超过 4 卡，一律拒绝启动，绝不下探到模型加载才发现问题。
    #
    # 两种设备来源都支持（F-1）：宿主侧是显式卡号（静态判 6/7）；容器侧被
    # nvidia-container-runtime 收窄后，`NVIDIA_VISIBLE_DEVICES` 变成哨兵 `void`，
    # 此时按**容器内实测可见卡**断言（`nvidia-smi -L`），"不得超 4 卡/不得为空"
    # 仍然成立。旧版把 `void` 当非法 ⇒ 目标 GPU 服务器上所有训练入口必然拒绝启动。
    devices = require_gpu_lock_or_exit()
    print(
        f"[train-worker] GPU lock OK: NVIDIA_VISIBLE_DEVICES={','.join(map(str, devices))}"
    )

    config = GraspoConfig.from_yaml(args.config)

    from graspo.flow.backend_selection import select_backend

    selection = select_backend(config)

    if config.train_method == "sft":
        from graspo.core.discovery import resolve_backend_builder

        # 注册表解析（SFT 专用），与 RL 分开——SFT 训练器形状不同于 RL 训练器。
        builder = resolve_backend_builder(selection.name, train_method="sft")
        builder(config, selection).train(smoke=args.smoke)
    else:
        from graspo.flow.backend_selection import create_trainer

        create_trainer(config, selection).train(smoke=args.smoke)


if __name__ == "__main__":
    main()
