#!/usr/bin/env bash
# Build GRASPO Docker image. Run from repository root.
#   IMAGE_NAME=graspo:test bash docker/build.sh
#   PROXY=http://127.0.0.1:18080 bash docker/build.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="${VERSION:-$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")}"
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