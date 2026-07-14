#!/usr/bin/env bash
# Build GRASPO Docker image. Run from repository root.
#   IMAGE_NAME=graspo:test bash docker/build.sh
#   VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")
#   IMAGE_NAME="${IMAGE_NAME:-graspo:${VERSION}}"
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Version is derived from the latest git tag (setuptools-scm compatible).
# Falls back to "0.0.0" if no tags exist (e.g. fresh clone for development).
VERSION=$(git -C "$ROOT_DIR" describe --tags --abbrev=0 2>/dev/null || echo "0.0.0")
IMAGE_NAME="${IMAGE_NAME:-graspo:${VERSION}}"

echo "Building ${IMAGE_NAME} from ${ROOT_DIR} (version=${VERSION})"
docker build \
  --build-arg VERSION="${VERSION}" \
  -t "${IMAGE_NAME}" \
  -f "${ROOT_DIR}/docker/Dockerfile" \
  "${ROOT_DIR}"

echo "Built ${IMAGE_NAME}"