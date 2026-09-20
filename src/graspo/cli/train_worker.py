"""训练 worker 进程入口：按 train_method + backend 分派训练器（支持 --smoke）。

**分派真相源**：``core.discovery.resolve_backend_builder`` —— 它按
``_REGISTRY_BY_TRAIN_METHOD`` 的**路由表**把 ``(train_method, backend)`` 解析成
训练器工厂，四种训练方法各查自己的注册表
（``graspo.backends`` / ``graspo.sft_backends`` / ``graspo.cpt_backends`` /
``graspo.opd_backends``），都在 ``pyproject.toml`` 的 entry_points 里声明，
开发模式下回退到 ``_DEV_FALLBACKS``。

**本文件不决定"哪个算法走哪张表"**——路由归属路由表（宪法 §1.1 模块边界 /
§1.4 单一真相源）。因此新增训练方法（如 CPT / OPD）与新增后端都**不需要改动
本文件**的任何分支（§1.2 对扩展开放、对修改关闭；决策 D5）。

**形状契约**：每张注册表里的工厂都满足
``factory(config, selection) -> 含 train(smoke: bool) 的训练器``。
native SFT 由 ``create_native_sft_trainer`` 提供，msswift SFT 由
``create_msswift_sft_trainer``（延迟构造器）提供；RL / CPT / OPD 各自的工厂
提供同一契约（都返回含 ``train(smoke)`` 的训练器）。
"""

import argparse

from graspo.core.discovery import resolve_backend_builder
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

    from graspo.flow.logging import run_log_id, set_run_id

    # 日志目录身份来自 config（§7.1/§10.1）：`training.run_name` 是 config 自带的
    # 运行标识，每个 rank 读同一份 YAML ⇒ 天然一致，不再需要环境变量在进程间
    # 传递（旧的 GRASPO_RUN_ID 通道已删除，见 flow/logging.get_run_id）。
    # `run_log_id` 只做一次格式归一（`graspo_<时间戳>` → `<YYYYMMDD-HHMMSS>`），
    # 不引入第二个来源——它保证各 rank 得到**同一个**目录名，而不是各按自己的
    # 时钟生成（那会把同一 launch 的日志拆进多个目录）。
    # 必须在任何 run_log_dir/setup_logging 之前绑定（单一真相源，§1.4）。
    set_run_id(run_log_id(config.training.run_name))

    from graspo.flow.backend_selection import select_backend

    selection = select_backend(config)

    # 统一路由：`(train_method, backend)` → 训练器工厂，四种训练方法走同一条路径。
    # 旧版是 `if train_method == "sft": ... else: <RL>`——它把 CPT/OPD 静默送进 RL
    # 注册表（拿着另一种算法去训练）。路由回归路由表后，本文件不再按算法分支。
    builder = resolve_backend_builder(selection.name, train_method=config.train_method)
    builder(config, selection).train(smoke=args.smoke)


if __name__ == "__main__":
    main()
