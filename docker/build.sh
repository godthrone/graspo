#!/usr/bin/env bash
# Build GRASPO Docker image. Run from repository root.
#   VERSION=v0.23.0 bash docker/build.sh        # 无 git tag 的环境必须显式传 VERSION
#   IMAGE_NAME=graspo:test bash docker/build.sh
#   HTTP_PROXY=http://proxy.example.com:8080 bash docker/build.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${VERSION:-}" ]; then
    VERSION="$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")"
    if [ "$VERSION" = "0.0.0" ]; then
        echo "WARNING: 无法从 git tag 推导版本号，使用默认版本 0.0.0" >&2
    fi
fi
IMAGE_NAME="${IMAGE_NAME:-graspo:${VERSION}}"
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"
NO_PROXY="${NO_PROXY:-}"

echo "Building ${IMAGE_NAME} from ${ROOT_DIR} (version=${VERSION})"
docker build \
  --network=host \
  --build-arg VERSION="${VERSION}" \
  --build-arg HTTP_PROXY="${HTTP_PROXY}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY}" \
  --build-arg NO_PROXY="${NO_PROXY}" \
  -t "${IMAGE_NAME}" \
  -f "${ROOT_DIR}/docker/Dockerfile" \
  "${ROOT_DIR}"

echo "Built ${IMAGE_NAME}"