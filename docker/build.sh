#!/usr/bin/env bash
# Build GRASPO Docker image. Run from repository root.
#   IMAGE_NAME=graspo:0.7.0 bash docker/build.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION=$(python -c "import tomllib; print(tomllib.load(open('$ROOT_DIR/pyproject.toml','rb'))['project']['version'])")
IMAGE_NAME="${IMAGE_NAME:-graspo:${VERSION}}"

echo "Building ${IMAGE_NAME} from ${ROOT_DIR}"
docker build -t "${IMAGE_NAME}" -f "${ROOT_DIR}/docker/Dockerfile" "${ROOT_DIR}"

echo "Built ${IMAGE_NAME}"