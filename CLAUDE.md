# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Amphion is an open-source toolkit for Audio, Music, and Speech Generation. It supports TTS, Voice Conversion (VC), Singing Voice Conversion (SVC), Text-to-Audio (TTA), and neural audio codecs. Key models include MaskGCT (ICLR 2025), Vevo, DualCodec, and Metis.

## Environment Setup

```bash
# System dependency (required for phonemizer)
sudo apt-get install espeak-ng

# Full environment (older models, uses Python 3.9.15)
conda create -n amphion python=3.9.15
conda activate amphion
sh env.sh

# MaskGCT (recommended Python 3.10)
conda create -n maskgct python=3.10
pip install -r models/tts/maskgct/requirements.txt

# Vevo (recommended Python 3.10)
conda create -n vevo python=3.10
pip install -r models/vc/vevo/requirements.txt

# DualCodec (pip-installable)
pip install dualcodec
```

The monotonic_align module requires compilation before use:
```bash
cd modules/monotonic_align && python setup.py build_ext --inplace
```

## Common Commands

### Training (all models use HuggingFace Accelerate)
```bash
export PYTHONPATH=$(pwd)

# Standard pattern for older models (TTS/SVC/Vocoder)
sh egs/{task}/{ModelName}/run.sh --stage 2 --config egs/{task}/{ModelName}/exp_config.json --name my_experiment --gpu 0

# Vevo (VC) training
CUDA_VISIBLE_DEVICES="0" accelerate launch --mixed_precision="bf16" \
    bins/vc/train.py --config=egs/vc/AutoregressiveTransformer/ar_conversion.json --exp_name=ar_conversion

# Noro training
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26667 --mixed_precision fp16 \
    bins/vc/train.py --config $config --exp_name $exp_name
```

### Preprocessing
```bash
# Stage 1 in run.sh scripts
CUDA_VISIBLE_DEVICES=0 python bins/{task}/preprocess.py --config=exp_config.json --num_workers=4
```

### Inference
```bash
# MaskGCT
python -m models.tts.maskgct.maskgct_inference

# Vevo variants
python -m models.vc.vevo.infer_vevotimbre
python -m models.vc.vevo.infer_vevostyle
python -m models.vc.vevo.infer_vevovoice
python -m models.vc.vevo.infer_vevotts

# MaskGCT Gradio demo
python -m models.tts.maskgct.gradio_demo

# Standard models (via run.sh stage 3)
sh egs/{task}/{ModelName}/run.sh --stage 3 --infer_expt_dir ckpts/{exp_name} ...
```

### Code Formatting
```bash
# CI enforces Black formatting on push/PR
black --check --diff .
black .  # to auto-format
```

## Architecture

### Execution Flow
Each model follows a 3-stage pipeline: **preprocess** -> **train** -> **inference**. Entry points live in `bins/{task}/` (train.py, inference.py, preprocess.py).

### Trainer Hierarchy
```
models/base/new_trainer.py (BaseTrainer - Accelerate-based)
├── models/tts/base/tts_trainer.py (TTSTrainer)
│   ├── FastSpeech2Trainer, VITSTrainer, VALLETrainer, JetsTrainer, NS2Trainer
├── models/vc/vevo/ (ARTrainer, FMTTrainer)
├── models/svc/base/ (SVCTrainer)
└── models/vocoders/ (VocoderTrainer)
```

Each trainer overrides: `_build_model()`, `_build_dataset()`, `_build_criterion()`, `_train_step()`.

### Configuration System
- `config/base.json` - global defaults (~200 parameters for preprocessing, model, training)
- `config/{model}.json` - model-specific overrides (e.g., `vits.json`, `valle.json`)
- `egs/{task}/{Model}/exp_config.json` - experiment configs that reference a `base_config` and override dataset paths, hyperparameters

Configs use JSON with comment support (`//`). Experiment configs inherit from base via `"base_config": "config/xxx.json"`.

### Module Organization
- `models/` - Complete model implementations (architecture + trainer + dataset + inference)
- `modules/` - Reusable neural building blocks (transformers, flows, diffusion, vocoders)
- `bins/` - CLI entry points dispatching to model trainers
- `egs/` - Recipes with run scripts and experiment configs
- `preprocessors/` - Per-dataset preprocessing (30+ datasets: LibriTTS, LJSpeech, VCTK, M4Singer, etc.)
- `evaluation/` - Objective metrics (F0, speaker similarity, intelligibility, spectral)
- `utils/` - Audio I/O, mel extraction, F0, tokenization, config loading

### Newer Models (Self-Contained)
MaskGCT, Vevo, DualCodec, and Metis are more self-contained with their own requirements.txt, inference scripts, and HuggingFace model downloads. They use `python -m models.{task}.{model}.{script}` invocation pattern instead of the `bins/` + `egs/` recipe system.

### Key Patterns
- Models download pretrained weights from HuggingFace (`amphion/` namespace)
- Training outputs go to `ckpts/{exp_name}/`
- Preprocessed data goes to the `processed_dir` specified in config (default: `data/`)
- All run.sh scripts set `PYTHONPATH` to the repo root and use `accelerate launch`
- The `--main_process_port` flag avoids port conflicts in multi-experiment setups

## Datasets
Training data primarily uses the Emilia dataset (101k+ hours, 6 languages). Older models support standard benchmarks (LibriTTS, LJSpeech, VCTK, etc.) via preprocessors in `preprocessors/{dataset_name}.py`.

## Files Ignored by Git
Model checkpoints (*.pt, *.ckpt), audio files (*.wav, *.flac), numpy files (*.npy), the `data/`, `ckpts/`, and `pretrained/wenet/` directories are all gitignored.
