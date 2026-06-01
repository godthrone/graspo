# Docker

Docker 镜像基于：

```text
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

构建：

```bash
bash scripts/build_docker.sh
```

覆盖镜像名：

```bash
IMAGE_NAME=graspo:dev bash scripts/build_docker.sh
```

镜像只包含代码和 Python 依赖。模型、数据、配置和输出目录都由 `scripts/run_train.sh` 在运行时挂载。

