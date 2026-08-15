#!/usr/bin/env bash
# Build GRASPO Docker image. Run from repository root.
#   VERSION=v0.23.0 bash docker/build.sh        # 无 git tag 的环境必须显式传 VERSION
#   IMAGE_NAME=graspo:test bash docker/build.sh
#   PROXY=http://proxy.example.com:8080 bash docker/build.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 版本号防呆（宪法 §2）：git tag 推导失败绝不静默回退 0.0.0——
# 那会以未知版本号覆盖已有镜像。必须显式传 VERSION。
if [ -z "${VERSION:-}" ]; then
    VERSION="$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || true)"
    if [ -z "${VERSION}" ]; then
        echo "ERROR: 无法从 git tag 推导版本号（当前目录无 .git 或没有 tag）。" >&2
        echo "      请显式指定版本：VERSION=v0.XX.0 bash docker/build.sh" >&2
        echo "      显式 VERSION 在无 git 环境下是唯一安全的构建方式。" >&2
        exit 1
    fi
fi
IMAGE_NAME="${IMAGE_NAME:-graspo:${VERSION}}"
PROXY="${PROXY:-}"

echo "Building ${IMAGE_NAME} from ${ROOT_DIR} (version=${VERSION})"
if [ -n "${PROXY}" ]; then
    echo "Using proxy: ${PROXY}"
    docker build \
      --network=host \
      --build-arg VERSION="${VERSION}" \
      --build-arg HTTP_PROXY="${PROXY}" \
      --build-arg HTTPS_PROXY="${PROXY}" \
      -t "${IMAGE_NAME}" \
      -f "${ROOT_DIR}/docker/Dockerfile" \
      "${ROOT_DIR}"
else
    docker build \
      --network=host \
      --build-arg VERSION="${VERSION}" \
      -t "${IMAGE_NAME}" \
      -f "${ROOT_DIR}/docker/Dockerfile" \
      "${ROOT_DIR}"
fi

echo "Built ${IMAGE_NAME}"