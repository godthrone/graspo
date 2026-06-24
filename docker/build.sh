#!/usr/bin/env bash
# Build GRASPO Docker image. Run from repository root.
#   IMAGE_NAME=graspo:0.7.0 bash docker/build.sh
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-graspo:0.7.0}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building ${IMAGE_NAME} from ${ROOT_DIR}"
docker build -t "${IMAGE_NAME}" -f "${ROOT_DIR}/docker/Dockerfile" "${ROOT_DIR}"

echo "Built ${IMAGE_NAME}"