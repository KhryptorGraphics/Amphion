# Docker Quickstart Guide

This guide covers running Amphion with Docker — from a single-command inference run to
a full local web stack with live code reloading.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Inference with Pre-Built Image](#2-quick-inference-with-pre-built-image)
3. [Full Web UI via docker-compose](#3-full-web-ui-via-docker-compose)
4. [Building Images Locally](#4-building-images-locally)
5. [Environment Variables Reference](#5-environment-variables-reference)
6. [Volume Mounts](#6-volume-mounts)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

### Docker Engine or Docker Desktop

| Platform | Installation |
|----------|-------------|
| Linux | [Docker Engine](https://docs.docker.com/engine/install/) |
| macOS | [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/) |
| Windows | [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) |

Minimum recommended versions: **Docker Engine 25+** or **Docker Desktop 4.28+**.

### NVIDIA Driver

A compatible NVIDIA driver must be installed on the host machine:

```bash
# Verify your driver and CUDA version
nvidia-smi
```

The images are built against **CUDA 12.4** (Ubuntu 22.04 base).  CUDA 12.x drivers are
backward-compatible, so any NVIDIA driver ≥ 525.60 should work.

### nvidia-container-toolkit

The nvidia-container-toolkit allows Docker containers to access your GPU:

```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Full instructions: [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Verify GPU Access in Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see the `nvidia-smi` output with your GPU listed.

---

## 2. Quick Inference with Pre-Built Image

Pre-built images are published to Docker Hub under `realamphion/amphion`.

### Available Image Tags

| Tag | Description | Approx. Size |
|-----|-------------|-------------|
| `realamphion/amphion:inference` | Minimal inference environment (torch, transformers, gradio) | ~5 GB |
| `realamphion/amphion:training` | Full training stack (adds tensorboard, fairseq, codecs, etc.) | ~12 GB |
| `realamphion/amphion:webui` | Web UI layer (adds Node.js, FastAPI, built React frontend) | ~14 GB |

### Pull the inference image

```bash
docker pull realamphion/amphion:inference
```

### MaskGCT — Zero-Shot TTS

```bash
# Clone the repo so models/ and egs/ are available at runtime
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion

# Run MaskGCT inference — GPU required
docker run --rm --gpus all \
    -v "$(pwd)":/app \
    -v hf_cache:/root/.cache/huggingface \
    -v "$(pwd)/output":/app/output \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/app \
    -w /app \
    realamphion/amphion:inference \
    conda run --no-capture-output -n amphion \
        python -m models.tts.maskgct.maskgct_inference
```

Generated audio files are written to `./output/` on your host.

### Vevo — Zero-Shot Voice Conversion

```bash
# Vevo timbre conversion
docker run --rm --gpus all \
    -v "$(pwd)":/app \
    -v hf_cache:/root/.cache/huggingface \
    -v "$(pwd)/output":/app/output \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/app \
    -w /app \
    realamphion/amphion:inference \
    conda run --no-capture-output -n amphion \
        python -m models.vc.vevo.infer_vevotimbre

# Vevo style / accent conversion
docker run --rm --gpus all \
    -v "$(pwd)":/app \
    -v hf_cache:/root/.cache/huggingface \
    -v "$(pwd)/output":/app/output \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/app \
    -w /app \
    realamphion/amphion:inference \
    conda run --no-capture-output -n amphion \
        python -m models.vc.vevo.infer_vevostyle
```

### MaskGCT Gradio Demo

Launch the interactive Gradio demo and open it in your browser:

```bash
docker run --rm --gpus all \
    -p 14557:14557 \
    -v "$(pwd)":/app \
    -v hf_cache:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/app \
    -w /app \
    realamphion/amphion:inference \
    conda run --no-capture-output -n amphion \
        python -m models.tts.maskgct.gradio_demo
```

Then open http://localhost:14557 in your browser.

### Interactive Shell

Drop into a bash shell inside the container for ad-hoc exploration:

```bash
docker run --rm --gpus all -it \
    -v "$(pwd)":/app \
    -v hf_cache:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/app \
    -w /app \
    realamphion/amphion:inference
```

---

## 3. Full Web UI via docker-compose

The docker-compose stack launches the FastAPI backend (with the pre-built React frontend)
using a single command.  An NVIDIA GPU and nvidia-container-toolkit are still required.

### Production Mode (API + bundled frontend)

```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion

# Pull the latest webui image and start the stack
docker compose pull
docker compose up -d
```

The API (and bundled React UI) are available at **http://localhost:14555**.

| Endpoint | URL |
|----------|-----|
| Web UI | http://localhost:14555 |
| REST API | http://localhost:14555/api/ |
| API Docs (Swagger) | http://localhost:14555/api/docs |
| Health check | http://localhost:14555/api/health |

To stop:

```bash
docker compose down
```

### Optional: Frontend Dev Server (hot-reload)

For active frontend development, start the optional `frontend` service that runs the
Next.js dev server with hot-reload on port 14556:

```bash
docker compose --profile frontend up
```

| Service | URL |
|---------|-----|
| API (production build) | http://localhost:14555 |
| Frontend dev server | http://localhost:14556 |

### Development Mode (live Python reloading)

The repository ships a `docker-compose.override.yml` that is automatically merged by
Docker Compose.  It bind-mounts the Python source tree and enables uvicorn `--reload`:

```bash
# Override file is applied automatically — no extra flags needed
docker compose up
```

Changes you make to `.py` files under `models/`, `modules/`, `bins/`, `utils/`, etc.
are immediately reflected in the running container.

> **Note:** `docker-compose.override.yml` is intended for local development only.
> In CI/CD or production, copy only `docker-compose.yml`.

### Checking Service Health

```bash
# Check container status and health
docker compose ps

# Follow API logs
docker compose logs -f api

# Follow frontend logs
docker compose logs -f frontend
```

---

## 4. Building Images Locally

The Dockerfile is a **multi-stage build** with five named stages.  Use `--target` to
build only the layers you need.

### Stage Overview

| Target | Extends | Contents |
|--------|---------|----------|
| `base` | — | Ubuntu 22.04 + CUDA 12.4 + system packages |
| `python-base` | `base` | Miniconda + Python 3.10 `amphion` conda env |
| `inference` | `python-base` | torch 2.0.1, transformers, gradio, phonemizer, … |
| `training` | `inference` | + tensorboard, fairseq, codecs, evaluation tools |
| `webui` | `training` | + Node.js 20, FastAPI/uvicorn, built React frontend |

### Build Commands

```bash
# Inference-only image (~5 GB, fastest to build)
docker build --target inference -t realamphion/amphion:inference .

# Full training image
docker build --target training -t realamphion/amphion:training .

# Web UI image (API + bundled React)
docker build --target webui -t realamphion/amphion:webui .

# Base CUDA + Python environment only
docker build --target python-base -t realamphion/amphion:python-base .
```

### Using Build Cache

Docker layer caching means incremental rebuilds are fast.  If you only change Python
packages in the `training` stage, only that layer and above are rebuilt.

```bash
# Force a clean rebuild (no cache)
docker build --no-cache --target inference -t realamphion/amphion:inference .
```

### Multi-Platform Builds (experimental)

```bash
# Build for both amd64 and arm64 (requires buildx)
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --target inference \
    -t realamphion/amphion:inference \
    --push .
```

> **Note:** GPU acceleration is only available on `linux/amd64` with an NVIDIA GPU.
> The `arm64` build produces a CPU-only image.

---

## 5. Environment Variables Reference

These variables can be passed to `docker run -e` or added to the `environment:` section
of `docker-compose.yml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | *(unset)* | Must be set to `/app` so Python can find the Amphion modules. Always set this. |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory. Set to a bind-mounted path to persist downloaded model weights across container restarts. |
| `OUTPUT_DIR` | `/app/output` | Directory where generated audio files are written by inference scripts. Mount a host path here to retrieve results. |
| `CUDA_VISIBLE_DEVICES` | *(all GPUs)* | Restrict which GPUs the container can see. Example: `CUDA_VISIBLE_DEVICES=0` for the first GPU only. |
| `NCCL_DEBUG` | *(unset)* | Set to `INFO` or `WARN` to enable NCCL communication debug logs (useful for multi-GPU training issues). |
| `DEBUG` | `false` | Set to `true` to enable verbose FastAPI / uvicorn debug logging (used by `docker-compose.override.yml`). |
| `LOG_LEVEL` | `info` | uvicorn log level: `debug`, `info`, `warning`, `error`, `critical`. |
| `PYTHONDONTWRITEBYTECODE` | `1` | Prevents Python from writing `.pyc` files inside bind-mounted source directories. |
| `PYTHONUNBUFFERED` | `1` | Ensures Python stdout/stderr appear immediately in `docker logs`. |
| `BUILD_ENV` | `production` | Build-time ARG passed to `docker compose build`. Set to `development` via `docker-compose.override.yml`. |
| `PORT` | `14556` | Port the Next.js dev server listens on (frontend service only). |
| `REACT_APP_API_BASE_URL` | `http://api:14555` | Base URL the React app uses to reach the FastAPI backend (frontend service only). |

### Example: Custom HuggingFace Cache Location

```bash
docker run --rm --gpus all \
    -v "$(pwd)":/app \
    -v /data/hf_models:/hf_cache \
    -e HF_HOME=/hf_cache \
    -e PYTHONPATH=/app \
    -w /app \
    realamphion/amphion:inference \
    conda run --no-capture-output -n amphion \
        python -m models.tts.maskgct.maskgct_inference
```

---

## 6. Volume Mounts

Mounting the right directories ensures model weights persist across container restarts
and generated outputs are accessible on your host.

### Recommended Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./` (repo root) | `/app` | Amphion source code and experiment configs |
| `/data/hf_cache` or named volume | `/root/.cache/huggingface` (or `$HF_HOME`) | HuggingFace model weights (persists downloads) |
| `./output` | `/app/output` | Generated audio files |
| `./ckpts` or named volume | `/app/ckpts` | Training checkpoints |
| `./pretrained` or named volume | `/app/pretrained` | Pretrained model weights (non-HuggingFace) |
| `./data` | `/app/data` | Preprocessed training data |

### Named Volumes vs. Bind Mounts

- **Named volumes** (`docker volume`) are managed by Docker, survive container removal,
  and are the recommended choice for model weights and checkpoints that do not need
  direct host access.

- **Bind mounts** (host paths like `./output`) give you direct access to generated
  files from your host OS — use these for output directories.

The `docker-compose.yml` uses named volumes for `pretrained` and `ckpts`, and a bind
mount for `./output`:

```yaml
volumes:
  - ./output:/app/output          # bind mount — easy host access to generated files
  - pretrained:/app/pretrained    # named volume — persists HF model weights
  - ckpts:/app/ckpts              # named volume — persists training checkpoints
```

### Inspect or Backup a Named Volume

```bash
# List volumes
docker volume ls

# Inspect the pretrained model cache
docker volume inspect amphion_pretrained

# Copy files out of a named volume
docker run --rm -v amphion_pretrained:/data -v "$(pwd)":/backup \
    busybox tar czf /backup/pretrained_backup.tar.gz -C /data .
```

---

## 7. Troubleshooting

### GPU Not Detected Inside Container

**Symptom:** `nvidia-smi` inside the container fails, or PyTorch reports no CUDA device.

**Checks:**

```bash
# 1. Confirm nvidia-container-toolkit is installed and Docker runtime is configured
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Verify GPU access outside of compose
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 3. Ensure the compose file has the GPU reservation block
grep -A5 'deploy' docker-compose.yml
```

The correct GPU reservation block in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

> **Note:** `docker compose` (V2 plugin) supports `deploy.resources.reservations`.
> The older `docker-compose` (V1, Python) may require `--compatibility` flag.

---

### Out-of-Memory (OOM) Errors

**Symptom:** Container is killed with exit code 137, or PyTorch raises `CUDA out of memory`.

**Solutions:**

```bash
# 1. Check how much VRAM is free on the host
nvidia-smi

# 2. Limit container to a single GPU
docker run --rm --gpus '"device=0"' ...

# 3. Set a smaller batch size or reduce diffusion steps in your inference config

# 4. Enable memory-efficient attention (if supported by the model)
# Edit your inference script and add:
#   torch.backends.cuda.enable_flash_sdp(True)
```

For training, reduce `batch_size` in your `exp_config.json` and re-run.

---

### Container Starts But API Immediately Exits

**Symptom:** `docker compose ps` shows the `api` service as `Exited`.

**Diagnosis:**

```bash
# View the last 100 lines of API logs
docker compose logs --tail=100 api
```

Common causes:

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'models'` | PYTHONPATH is not set. Add `-e PYTHONPATH=/app` or check `environment:` in compose. |
| `Port 14555 already in use` | Another process is using port 14555. Run `sudo lsof -i :14555` and stop the conflicting process. |
| `No such file or directory: '/app/models/web/api/main.py'` | The source tree is not mounted. Add `-v "$(pwd)":/app` or check `volumes:` in compose. |
| CUDA driver/library version mismatch | Ensure the host NVIDIA driver is ≥ 525 for CUDA 12.4 images. |

---

### Image Build Fails on `npm ci`

**Symptom:** The `webui` stage fails with `npm ERR! code ENOENT` or lock file errors.

**Fix:** Ensure `models/web/react/package-lock.json` is committed to the repo and not
listed in `.dockerignore`.  Then rebuild:

```bash
docker build --no-cache --target webui -t realamphion/amphion:webui .
```

---

### Slow Model Downloads on First Run

HuggingFace models (MaskGCT weights ~3 GB, Vevo ~2 GB) are downloaded on the first
inference run.  To avoid re-downloading across container restarts, always mount a
persistent cache:

```bash
# Use a named volume
docker volume create hf_cache
docker run --rm --gpus all \
    -v hf_cache:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    ...
```

---

### Permission Errors on Mounted Directories

**Symptom:** `PermissionError: [Errno 13] Permission denied: '/app/output/...'`

**Fix:** Ensure the host output directory exists and is writable before running:

```bash
mkdir -p output && chmod 777 output
docker run --rm --gpus all -v "$(pwd)/output":/app/output ...
```

Alternatively, specify `--user "$(id -u):$(id -g)"` to run as your host user:

```bash
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -v "$(pwd)":/app \
    -v "$(pwd)/output":/app/output \
    realamphion/amphion:inference ...
```

---

### Further Resources

- [Amphion README](../README.md)
- [Deployment Guide](../DEPLOYMENT.md)
- [Port Configuration](../PORT_CONFIGURATION.md)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [GitHub Container Registry (GHCR)](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
