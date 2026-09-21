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
        "--determinism-spec",
        default="",
        help=(
            "Internal (graspo launch --determinism) JSON declaration of the determinism "
            "switches to pin in this process. Empty = all off (default): nothing is "
            "injected, nothing is printed, no artifact is written."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke boundary (infrastructure param): run until the first optimize"
            " step then stop. Never mutates the config object."
        ),
    )
    args = parser.parse_args()

    # ★ 确定性开关的**环境变量半场**必须在这里落地 —— 这是本进程最早的、同时还能
    #   读到 `--determinism-spec` 的位置（`train_worker` 的模块级导入实测不碰 torch，
    #   所以此刻 torch / NCCL 都还没初始化）。为什么这一步必不可少：矩阵 runner 路线
    #   （`entry.sh`）只把 spec 交给 worker，**没有父进程注入环境**这一步 ⇒ 改前
    #   环境变量半场在容器里一条都没生效（只被打印、被记录）。详见
    #   `core/determinism.py` 模块 docstring 的"两半场"与 `apply_env_determinism`。
    from graspo.core.determinism import (
        DeterminismSwitch,
        apply_env_determinism,
        bind_active_switch,
    )

    try:
        # 绑定即唯一真相源（§1.4）：训练层只查 `active_switch()`，不各自解析 spec。
        # §2.3 边界校验即防呆：非法声明 fail-closed，不得静默当"关"；且天然早于
        # 锁卡守卫 —— 连卡都还没碰就先拒绝，方向只会更安全。
        determinism = bind_active_switch(DeterminismSwitch.from_spec(args.determinism_spec))
    except ValueError as exc:
        raise SystemExit(f"--determinism-spec 非法：{exc}") from exc
    # 默认关 ⇒ 空 dict ⇒ 一行都不打、os.environ 一字不改（零变化，§2.2）。
    for env_name, env_value in apply_env_determinism(determinism).items():
        print(f"[train-worker] determinism env: {env_name}={env_value}")

    # 锁卡守卫（fail-closed）：训练进程启动的第一件事。未显式锁卡 / 含生产卡
    # GPU6,7 / 超过 4 卡，一律拒绝启动，绝不下探到模型加载才发现问题。
    #
    # 两种设备来源都支持（F-1）：宿主侧是显式卡号（静态判 6/7）；容器侧被
    # nvidia-container-runtime 收窄后，`NVIDIA_VISIBLE_DEVICES` 变成哨兵 `void`，
    # 此时按**容器内实测可见卡**断言（`nvidia-smi -L`），"不得超 4 卡/不得为空"
    # 仍然成立。旧版把 `void` 当非法 ⇒ 目标 GPU 服务器上所有训练入口必然拒绝启动。
    devices = require_gpu_lock_or_exit()
    print(f"[train-worker] GPU lock OK: NVIDIA_VISIBLE_DEVICES={','.join(map(str, devices))}")

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

    # 确定性钉定开关的**进程内 torch API 半场**（环境变量半场已在 `main()` 开头由
    # `apply_env_determinism` 落地）。位置刻意在锁卡守卫与 set_run_id **之后**、
    # 训练器构造 **之前**：守卫的先执行属性与日志身份都不被这条改动影响。
    # 默认（空 spec）时：`apply_torch_determinism` 返回空列表、
    # `format_determinism_banner` 返回空列表、`determinism_artifact` 返回 None
    # ⇒ 既不调用 torch、也不多打一行、不写任何文件（默认关零变化）。
    from graspo.core.determinism import (
        apply_torch_determinism,
        determinism_artifact,
        determinism_verify_artifact,
        format_determinism_banner,
        format_determinism_verify_report,
    )

    for line in format_determinism_banner(determinism):
        print(line)
    for statement in apply_torch_determinism(determinism):
        print(f"[train-worker] determinism applied: {statement}")
    # ★ "声明 vs 生效"回读核实（§2.2）：请求了钉定却没落下的项在这里**显式**报出，
    #   绝不静默通过（"开关开着但这些项没生效"正是本包要根治的形态）。
    for line in format_determinism_verify_report(determinism):
        print(line)
    # 记录进产物（§2.2）：与打印共用同一份渲染，事后核对不靠回忆。
    determinism_record = determinism_artifact(determinism)
    if determinism_record is not None:
        from graspo.flow.logging import append_jsonl_segment, run_log_dir

        artifact_path = run_log_dir(config.training.output_dir) / "determinism.jsonl"
        append_jsonl_segment(artifact_path, determinism_record)
        verify_record = determinism_verify_artifact(determinism)
        if verify_record is not None:
            # 第二行 = 回读结论：产物里同时有"请求了什么"与"回读到了什么"。
            append_jsonl_segment(artifact_path, verify_record)
        print(f"[train-worker] determinism artifact: {artifact_path}")

    from graspo.flow.backend_selection import select_backend

    selection = select_backend(config)

    # 统一路由：`(train_method, backend)` → 训练器工厂，四种训练方法走同一条路径。
    # 旧版是 `if train_method == "sft": ... else: <RL>`——它把 CPT/OPD 静默送进 RL
    # 注册表（拿着另一种算法去训练）。路由回归路由表后，本文件不再按算法分支。
    builder = resolve_backend_builder(selection.name, train_method=config.train_method)
    builder(config, selection).train(smoke=args.smoke)


if __name__ == "__main__":
    main()
