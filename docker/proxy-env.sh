#!/bin/sh
# 把 BuildKit secret 里的构建期代理读成环境变量（宪法 §14.3 取值规则）。
#
# 为什么是 secret 而不是 --build-arg：class A / 公开仓库若用 `--build-arg` 传内网代理，
# 取值会留在 `docker history --no-trunc` 里（即使从不写入 ENV）⇒ 必须走 `--mount=type=secret`。
#
# 用法（由 docker/Dockerfile 的每个需要网络的 RUN 引入）：
#     . /usr/local/bin/proxy-env.sh
#
# 语义：**无对应 secret ⇒ 该项不设置**（直连公网），不报错、不落任何默认值。
# 边界：只读 /run/secrets/*、只导出环境变量；不写文件、不打日志（代理取值不进构建日志）。
set -eu

if [ -s /run/secrets/http_proxy ]; then
    http_proxy="$(cat /run/secrets/http_proxy)"
    HTTP_PROXY="$http_proxy"
    export http_proxy HTTP_PROXY
fi

if [ -s /run/secrets/https_proxy ]; then
    https_proxy="$(cat /run/secrets/https_proxy)"
    HTTPS_PROXY="$https_proxy"
    export https_proxy HTTPS_PROXY
fi

if [ -s /run/secrets/no_proxy ]; then
    no_proxy="$(cat /run/secrets/no_proxy)"
    NO_PROXY="$no_proxy"
    export no_proxy NO_PROXY
fi
