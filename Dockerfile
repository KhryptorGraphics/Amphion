# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Available CUDA images: https://hub.docker.com/r/nvidia/cuda/tags

# ============================================================
# Stage 1: base — Ubuntu 22.04 + CUDA 12.4 + system packages
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV LANG=en_US.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    CONDA_HOME=/opt/conda

ENV PATH=$CONDA_HOME/bin:$CUDA_HOME/bin:$PATH \
    LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
    LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH \
    CONDA_PREFIX=$CONDA_HOME \
    NCCL_HOME=$CUDA_HOME

# Install system packages (no mirror substitutions needed on Ubuntu 22.04)
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
        espeak-ng \
        ffmpeg \
        git \
        less \
        wget \
        curl \
        libsm6 \
        libxext6 \
        libxrender-dev \
        build-essential \
        cmake \
        pkg-config \
        libx11-dev \
        libatlas-base-dev \
        libgtk-3-dev \
        libboost-python-dev \
        vim \
        libgl1 \
        libaio-dev \
        software-properties-common \
        tmux \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Stage 2: python-base — Miniconda + Python 3.10 environment
# ============================================================
FROM base AS python-base

ARG MINICONDA='Miniconda3-py310_23.3.1-0-Linux-x86_64.sh'

# Install Miniconda with Python 3.10
RUN wget -t 0 -c -O /tmp/anaconda.sh https://repo.anaconda.com/miniconda/${MINICONDA} \
    && /bin/bash /tmp/anaconda.sh -b -p $CONDA_HOME \
    && rm /tmp/anaconda.sh \
    && conda clean -afy

# Create amphion conda environment with Python 3.10
RUN conda create -y --name amphion python=3.10

RUN conda init \
    && echo "conda activate amphion" >> ~/.bashrc

WORKDIR /app

CMD ["/bin/bash"]

# ============================================================
# Stage 3: inference — minimal inference environment
# ============================================================
FROM python-base AS inference

# Install inference-only Python packages.
# Note: torch 2.0.1 ships cu117/cu118 wheels; cu118 is CUDA 12.x-compatible
# via CUDA backward-compatibility (no cu124 wheel exists for 2.0.1).
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        torch==2.0.1+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
    && conda run --no-capture-output -n amphion pip install --no-cache-dir \
        transformers==4.41.2 \
        accelerate==0.24.1 \
        numpy==1.26.0 \
        scipy==1.12.0 \
        librosa \
        encodec \
        phonemizer \
        g2p_en \
        pypinyin \
        tqdm \
        gradio

WORKDIR /app

CMD ["/bin/bash"]

# *** Build targets ***
# docker build --target python-base -t realamphion/amphion:python-base .
# docker build --target inference   -t realamphion/amphion:inference .
# docker build --target training    -t realamphion/amphion:training .
# docker build --target webui       -t realamphion/amphion:webui .

# *** Run ***
# cd Amphion
# docker run --runtime=nvidia --gpus all -it -v .:/app realamphion/amphion:inference

# *** Push and release ***
# docker login
# docker push realamphion/amphion
