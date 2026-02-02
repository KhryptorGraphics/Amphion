# Amphion CUDA 13 Environment Setup - COMPLETED

## Project Location
/home/kp/repo2/Amphion

## Environment Details
- **Conda env**: `amphion` (Python 3.12)
- **GPU**: NVIDIA Thor (Blackwell, SM 11.0)
- **CUDA**: 13.0
- **JetPack**: 7.2
- **Architecture**: aarch64 (ARM64)

## Key Software Versions
- PyTorch: 2.10.0+cu130 (built from source)
- torchaudio: Built from source (matching PyTorch)
- torchvision: Built from source (matching PyTorch)
- transformers, accelerate, diffusers: Latest pip versions
- onnxruntime: Built for aarch64 + CUDA 13

## Verified Models (GPU-accelerated)
1. MaskGCT (TTS) - models/tts/maskgct/
2. Vevo (Voice Conversion) - models/vc/vevo/
3. DualCodec (Neural Codec) - pip installed
4. Metis (uses monotonic_align)

## Compiled Modules
- monotonic_align: modules/monotonic_align/monotonic_align/core.cpython-312-aarch64-linux-gnu.so

## Environment Isolation
- PYTHONNOUSERSITE=1 (always set)
- pip config: no-user = true
- All packages in conda env only (NO system package mixing)

## Activation Pattern
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate amphion
export PYTHONPATH=/home/kp/repo2/Amphion
```

## Critical Learning
Blackwell architecture uses SM 11.0 (not SM 9.0 as some docs suggest). PyTorch correctly detects this as compute capability 11.0.

## Verification Commands
```bash
# Check PyTorch + CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Check GPU
nvidia-smi

# Test models
python -m models.tts.maskgct.maskgct_inference
python -m models.vc.vevo.infer_vevotimbre
```

## Beads Workflow
All tasks tracked under epic AMP-hzd (12 tasks total, all closed).
Use `bd list` to see task history.
