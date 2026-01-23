# Amphion: CUDA 13 / aarch64 / Blackwell Environment Setup

## Goal
Set up Amphion project with fully working conda environments on NVIDIA Thor (Blackwell/aarch64, CUDA 13.0, SM 9.0, JetPack 7.2). All CUDA-dependent packages must be compiled from source. Target latest Python version compatible with dependencies.

## Platform Specifications
- **Architecture**: aarch64 (ARM64)
- **GPU**: NVIDIA Thor (Blackwell, SM 9.0)
- **CUDA**: 13.0
- **JetPack**: 7.2
- **OS**: Linux (Tegra)

## Requirements

### Phase 1: Research & Compatibility Analysis
1. Determine latest Python version compatible with all Amphion dependencies (3.11 or 3.12)
2. Research PyTorch build requirements for CUDA 13 + aarch64
3. Identify which dependencies need source builds vs pip install
4. Check torchaudio/torchvision CUDA 13 compatibility
5. Research MaskGCT/Vevo/DualCodec specific dependency chains

### Phase 2: Base Environment Setup
1. Create conda environment with determined Python version
2. Install system dependencies (espeak-ng, ffmpeg, build tools)
3. Set environment isolation variables (PYTHONNOUSERSITE=1, etc.)
4. Configure CUDA_HOME, TORCH_CUDA_ARCH_LIST="9.0"

### Phase 3: PyTorch Stack (Source Build)
1. Build PyTorch from source for CUDA 13 + SM 9.0 + aarch64
2. Build torchaudio from source (matching PyTorch version)
3. Build torchvision from source (matching PyTorch version)
4. Verify CUDA detection and SM 9.0 support

### Phase 4: Audio/ML Dependencies
1. Install/build librosa, soundfile, audioread
2. Install/build transformers, accelerate, diffusers
3. Install/build encodec, vocos, speechtokenizer, descript-audio-codec
4. Install phonemizer + espeak-ng bindings
5. Install fairseq (may need source build for aarch64)

### Phase 5: Model-Specific Dependencies
1. MaskGCT requirements (models/tts/maskgct/requirements.txt)
2. Vevo requirements (models/vc/vevo/requirements.txt)
3. DualCodec (pip install dualcodec or source build)
4. Compile monotonic_align module
5. Metis-specific dependencies

### Phase 6: Verification & Testing
1. Verify torch.cuda.is_available() and correct SM architecture
2. Run MaskGCT inference test
3. Run Vevo inference test
4. Run DualCodec encode/decode test
5. Verify all model weights download from HuggingFace
6. Check GPU utilization during inference

## Constraints
- NEVER mix system and conda packages
- ALL CUDA packages built from source
- PYTHONNOUSERSITE=1 always active
- pip configured with no-user = true
- Verify isolation: which python/pip must point to $CONDA_PREFIX/bin/

## Status
- [x] Phase 1: Research (Python 3.12 + PyTorch 2.10.0+cu130 for SM 11.0)
- [x] Phase 2: Base Environment (amphion conda env with PYTHONNOUSERSITE=1)
- [x] Phase 3: PyTorch Stack (Built from source for CUDA 13.0 + aarch64)
- [x] Phase 4: Audio/ML Dependencies (All installed, including onnxruntime)
- [x] Phase 5: Model-Specific Dependencies (MaskGCT, Vevo, DualCodec, monotonic_align compiled)
- [x] Phase 6: Verification (All models GPU-accelerated, SM 11.0 confirmed)

## Completion Summary
All phases completed on 2026-01-23. Environment verified with:
- PyTorch 2.10.0+cu130 with CUDA 13.0
- NVIDIA Thor (SM 11.0) GPU detection working
- All models (MaskGCT, Vevo, DualCodec, Metis) tested on GPU
- Accelerate training loops validated
- FP16/BF16 autocast working
- monotonic_align Cython extension compiled for aarch64 Python 3.12

Note: Original PROMPT.md specified SM 9.0, but actual Blackwell architecture uses SM 11.0 (correctly detected and used).
