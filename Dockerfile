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

# ============================================================
# Stage 4: training — full training environment (extends inference)
# ============================================================
FROM inference AS training

# General utilities and data-processing packages
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        setuptools \
        ruamel.yaml \
        colorama \
        easydict \
        tabulate \
        loguru \
        json5 \
        Cython \
        unidecode \
        inflect \
        tgt \
        librosa==0.9.1 \
        matplotlib \
        typeguard \
        einops \
        omegaconf \
        hydra-core \
        humanfriendly \
        pandas \
        munch

# Training core: torchaudio/torchvision cu118 wheels + signal-processing libs
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        torchaudio==2.0.2+cu118 \
        torchvision==0.15.2+cu118 \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
    && conda run --no-capture-output -n amphion pip install --no-cache-dir \
        tensorboard \
        tensorboardX \
        diffusers \
        praat-parselmouth \
        audiomentations \
        pedalboard \
        ffmpeg-python==0.2.0 \
        pyworld \
        diffsptk==1.0.1 \
        nnAudio \
        ptwt

# Audio codec and generation packages
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        vocos \
        speechtokenizer \
        descript-audio-codec

# Evaluation and metrics packages
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        torchmetrics \
        pymcd \
        openai-whisper \
        frechet_audio_distance \
        asteroid \
        resemblyzer \
        vector-quantize-pytorch==1.12.5

# PESQ (from source — no stable PyPI wheel)
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        https://github.com/vBaiCai/python-pesq/archive/master.zip

# fairseq (installed separately due to complex dependencies)
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir fairseq

# lhotse (from git — requires unreleased features)
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        git+https://github.com/lhotse-speech/lhotse

# Pin phonemizer and pypinyin to versions tested with Amphion
RUN conda run --no-capture-output -n amphion pip install --no-cache-dir \
        phonemizer==3.2.1 \
        pypinyin==0.48.0

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
