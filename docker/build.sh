#!/usr/bin/env bash
# Build GRASPO Docker image. Run from repository root.
#   bash docker/build.sh                        # 版本从 git 版本 tag 推导
#   VERSION=v0.23.0 bash docker/build.sh        # 无 git tag 的环境必须显式传 VERSION
#   IMAGE_NAME=graspo:test bash docker/build.sh
#   HTTP_PROXY=http://proxy.example.com:8080 bash docker/build.sh
#   （代理按 §14.3 取值规则转成 BuildKit secret 传入，**不**走 --build-arg）
#   bash docker/build.sh --print-version        # 只打印推导出的版本并退出
#
# **版本号唯一来源（宪法 §1.4）**：本脚本的 `derive_version()` 是构建链路里
# 唯一推导版本的地方。`docker/Dockerfile*` 只接收 `--build-arg VERSION`；锁定
# uv.lock 时也必须复用同一来源，不要在两处各写一个版本号：
#   SETUPTOOLS_SCM_PRETEND_VERSION="$(bash docker/build.sh --print-version)" uv lock
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 只认匹配 [tool.setuptools_scm].tag_regex 的版本 tag（`v?X.Y.Z`）。
# 仓库里存在**故意不匹配**的基线 tag（如 `baseline-2026-09-17`）：`git describe`
# 会把它当最近 tag 返回，导致 `SETUPTOOLS_SCM_PRETEND_VERSION` 拿到非法版本、
# setuptools-scm 报 `Can't parse version from tag`。用 --match 过滤掉它。
derive_version() {
    local described
    described="$(git -C "$ROOT_DIR" describe --tags --abbrev=0 \
        --match 'v[0-9]*.[0-9]*.[0-9]*' 2>/dev/null || true)"
    if [ -n "$described" ]; then
        printf '%s' "$described"
        return 0
    fi
    local tagged
    tagged="$(git -C "$ROOT_DIR" tag --sort=-v:refname 2>/dev/null \
        | grep -E '^v?[0-9]+\.[0-9]+\.[0-9]+$' | head -1 || true)"
    if [ -n "$tagged" ]; then
        printf '%s' "$tagged"
        return 0
    fi
    printf '0.0.0'
}

if [ "${1:-}" = "--print-version" ]; then
    if [ -n "${VERSION:-}" ]; then
        printf '%s\n' "${VERSION}"
    else
        derive_version
        printf '\n'
    fi
    exit 0
fi

if [ -z "${VERSION:-}" ]; then
    VERSION="$(derive_version)"
    if [ "$VERSION" = "0.0.0" ]; then
        echo "WARNING: 无法从 git 版本 tag 推导版本号，使用默认版本 0.0.0" >&2
    fi
fi
IMAGE_NAME="${IMAGE_NAME:-graspo:${VERSION}}"
DOCKERFILE="${DOCKERFILE:-${ROOT_DIR}/docker/Dockerfile}"
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"
NO_PROXY="${NO_PROXY:-}"

# 代理只走 BuildKit secret（宪法 §14.3 取值规则）：class A / 公开仓库用 `--build-arg`
# 传代理会把取值留在 `docker history --no-trunc` 里 ⇒ 一律 `--secret id=<name>,env=<VAR>`。
# 宿主未设置（空值）⇒ 不传该 secret；Dockerfile 侧 `required=false` ⇒ 缺失即直连公网。
SECRET_ARGS=()
for pair in "http_proxy:HTTP_PROXY" "https_proxy:HTTPS_PROXY" "no_proxy:NO_PROXY"; do
    id="${pair%%:*}"
    var="${pair##*:}"
    if [ -n "${!var:-}" ]; then
        SECRET_ARGS+=(--secret "id=${id},env=${var}")
    fi
done

echo "Building ${IMAGE_NAME} from ${ROOT_DIR} (version=${VERSION}, dockerfile=${DOCKERFILE})"
docker build \
  --network=host \
  --build-arg VERSION="${VERSION}" \
  ${SECRET_ARGS[@]+"${SECRET_ARGS[@]}"} \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE}" \
  "${ROOT_DIR}"

echo "Built ${IMAGE_NAME}"
