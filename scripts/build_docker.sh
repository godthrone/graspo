#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-graspo:0.6.0-cuda13.2}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building ${IMAGE_NAME} from ${ROOT_DIR}"
docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"

echo "Built ${IMAGE_NAME}"

