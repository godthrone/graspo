#!/usr/bin/env bash
set -euo pipefail

TARGET_SERVER="${TARGET_SERVER:-}"
TARGET_PORT="${TARGET_PORT:-22}"
TARGET_PROJECT_DIR="${TARGET_PROJECT_DIR:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DRY_RUN="${DRY_RUN:-false}"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=accept-new}"

if [[ -z "${TARGET_SERVER}" ]]; then
  echo "ERROR: TARGET_SERVER is required, for example user@gpu-host" >&2
  exit 1
fi
if [[ -z "${TARGET_PROJECT_DIR}" ]]; then
  echo "ERROR: TARGET_PROJECT_DIR is required, for example /data/projects/graspo" >&2
  exit 1
fi

RSYNC_ARGS=(
  -av
  --delete
  --exclude .git
  --exclude .venv
  --exclude .idea
  --exclude __pycache__
  --exclude "*.egg-info"
  --exclude outputs
  --exclude checkpoints
  --exclude wandb
  --exclude runs
  --exclude logs
  --exclude models
  --exclude ".dataset"
  --exclude ".datasets"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

echo "Syncing ${PROJECT_DIR}/ -> ${TARGET_SERVER}:${TARGET_PROJECT_DIR}/"
if command -v rsync >/dev/null 2>&1; then
  rsync "${RSYNC_ARGS[@]}" -e "ssh -p ${TARGET_PORT} ${SSH_OPTS}" "${PROJECT_DIR}/" "${TARGET_SERVER}:${TARGET_PROJECT_DIR}/"
  exit 0
fi

echo "rsync not found; falling back to tar over ssh. The fallback does not delete stale remote files."
if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[DRY RUN] tar project and extract to ${TARGET_SERVER}:${TARGET_PROJECT_DIR}"
  exit 0
fi

tar -C "${PROJECT_DIR}" \
  --exclude .git \
  --exclude .venv \
  --exclude .idea \
  --exclude __pycache__ \
  --exclude "*.egg-info" \
  --exclude outputs \
  --exclude checkpoints \
  --exclude wandb \
  --exclude runs \
  --exclude logs \
  --exclude models \
  --exclude ".dataset" \
  --exclude ".datasets" \
  -czf - . | ssh -p "${TARGET_PORT}" ${SSH_OPTS} "${TARGET_SERVER}" \
    "mkdir -p '${TARGET_PROJECT_DIR}' && tar -xzf - -C '${TARGET_PROJECT_DIR}'"
