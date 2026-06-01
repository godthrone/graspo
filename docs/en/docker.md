# Docker

The Docker image is based on:

```text
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

Build:

```bash
bash scripts/build_docker.sh
```

Override the image name:

```bash
IMAGE_NAME=graspo:dev bash scripts/build_docker.sh
```

The image contains code and Python dependencies only. Models, data, configs, and
outputs are mounted at runtime by `scripts/run_train.sh`.

