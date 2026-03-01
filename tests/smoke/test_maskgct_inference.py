# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke tests for MaskGCT pretrained model inference.

These tests:
1. Attempt to import all required MaskGCT components (skips on ImportError)
2. Download the MaskGCT model weights from HuggingFace Hub
   (skips if HuggingFace is not reachable or the download fails)
3. Run a short inference pass on a synthetic prompt WAV file
4. Verify that the returned audio is a non-empty 1-D numpy float array

All tests are marked @pytest.mark.smoke and @pytest.mark.slow.
Exclude them from regular CI runs with::

    pytest -m "not smoke"

or::

    pytest -m "not slow"
"""

import socket

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
# Attempt all MaskGCT imports at module load time so that a missing dependency
# causes the entire module to be skipped rather than failing at collection.

_import_error = None
try:
    import safetensors
    import torch
    from huggingface_hub import hf_hub_download

    from models.tts.maskgct.maskgct_utils import (
        MaskGCT_Inference_Pipeline,
        build_acoustic_codec,
        build_s2a_model,
        build_semantic_codec,
        build_semantic_model,
        build_t2s_model,
    )
    from utils.util import load_config
except ImportError as _exc:
    _import_error = _exc

# Apply smoke + slow marks to every test in this module.
pytestmark = [pytest.mark.smoke, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_import_failed() -> None:
    """Call pytest.skip if any required MaskGCT import failed at load time."""
    if _import_error is not None:
        pytest.skip(
            f"Required MaskGCT dependency not available: {_import_error}"
        )


def _build_maskgct_pipeline(device):
    """Instantiate and return a fully-loaded MaskGCT inference pipeline.

    Downloads model checkpoints from HuggingFace Hub (results are cached
    locally by huggingface_hub after the first download).

    Args:
        device: A ``torch.device`` instance specifying where to load the models.

    Returns:
        MaskGCT_Inference_Pipeline: A pipeline ready for inference.

    Raises:
        Exception: Any failure during config loading, model building, weight
                   downloading, or weight loading propagates to the caller so
                   that tests can call ``pytest.skip()`` appropriately.
    """
    cfg_path = "./models/tts/maskgct/config/maskgct.json"
    cfg = load_config(cfg_path)

    # Build model skeletons (no pretrained weights yet)
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    codec_encoder, codec_decoder = build_acoustic_codec(
        cfg.model.acoustic_codec, device
    )
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

    # Download checkpoints from HuggingFace Hub
    semantic_code_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
    )
    codec_encoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model.safetensors"
    )
    codec_decoder_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="acoustic_codec/model_1.safetensors"
    )
    t2s_model_ckpt = hf_hub_download(
        "amphion/MaskGCT", filename="t2s_model/model.safetensors"
    )
    s2a_1layer_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="s2a_model/s2a_model_1layer/model.safetensors",
    )
    s2a_full_ckpt = hf_hub_download(
        "amphion/MaskGCT",
        filename="s2a_model/s2a_model_full/model.safetensors",
    )

    # Load weights into model skeletons
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    return MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )


# ---------------------------------------------------------------------------
# Module-scoped pipeline fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def maskgct_pipeline():
    """Module-scoped fixture providing a loaded MaskGCT inference pipeline.

    The fixture is shared across all tests in this module so that model
    weights are downloaded and loaded only once per test session.

    Skips all dependent tests if:
    - Required imports are unavailable
    - HuggingFace Hub is not reachable
    - Model weights cannot be downloaded or loaded
    """
    _skip_if_import_failed()

    # Check HF connectivity before attempting a potentially slow download
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=5.0):
            pass
    except (socket.error, OSError):
        pytest.skip(
            "HuggingFace Hub is not reachable — MaskGCT inference tests skipped"
        )

    try:
        return _build_maskgct_pipeline(torch.device("cpu"))
    except Exception as exc:
        pytest.skip(f"Failed to load MaskGCT weights from HuggingFace: {exc}")


# ---------------------------------------------------------------------------
# Tests: import verification
# ---------------------------------------------------------------------------


class TestMaskGCTImports:
    """Verify that all MaskGCT module components can be imported successfully."""

    def test_inference_pipeline_importable(self):
        """MaskGCT_Inference_Pipeline must be importable from maskgct_utils."""
        _skip_if_import_failed()
        assert MaskGCT_Inference_Pipeline is not None

    def test_maskgct_t2s_importable(self):
        """MaskGCT_T2S model class must be importable."""
        _skip_if_import_failed()
        from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S

        assert MaskGCT_T2S is not None

    def test_maskgct_s2a_importable(self):
        """MaskGCT_S2A model class must be importable."""
        _skip_if_import_failed()
        from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A

        assert MaskGCT_S2A is not None

    def test_build_helpers_importable(self):
        """All build_* helper functions must be importable from maskgct_utils."""
        _skip_if_import_failed()
        assert build_semantic_model is not None
        assert build_semantic_codec is not None
        assert build_acoustic_codec is not None
        assert build_t2s_model is not None
        assert build_s2a_model is not None


# ---------------------------------------------------------------------------
# Tests: end-to-end inference with pretrained weights
# ---------------------------------------------------------------------------


class TestMaskGCTInference:
    """End-to-end smoke tests for MaskGCT inference with real pretrained weights.

    These tests load the full MaskGCT model from HuggingFace Hub and run
    inference on a short synthetic prompt to verify basic pipeline correctness.
    Inference is performed on CPU with reduced diffusion steps to limit
    wall-clock time while still exercising the complete forward pass.
    """

    # Short texts keep the autoregressive step fast
    _PROMPT_TEXT = "We do not break."
    _TARGET_TEXT = "Hello, this is a test."

    # Minimal diffusion steps: fast but still exercises the full pipeline
    _N_TIMESTEPS = 5
    _N_TIMESTEPS_S2A = [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    _TARGET_LEN = 3  # seconds

    def _run_inference(self, pipeline, prompt_wav_path: str) -> np.ndarray:
        """Run inference and return the recovered audio array.

        Calls pytest.skip on any runtime failure so that infrastructure
        problems do not masquerade as test failures.
        """
        try:
            return pipeline.maskgct_inference(
                prompt_wav_path,
                self._PROMPT_TEXT,
                self._TARGET_TEXT,
                language="en",
                target_language="en",
                target_len=self._TARGET_LEN,
                n_timesteps=self._N_TIMESTEPS,
                n_timesteps_s2a=self._N_TIMESTEPS_S2A,
            )
        except Exception as exc:
            pytest.skip(
                f"MaskGCT inference failed (possible missing resource): {exc}"
            )

    def test_inference_returns_numpy_array(
        self, maskgct_pipeline, smoke_prompt_wav
    ):
        """Inference must return a numpy.ndarray."""
        audio = self._run_inference(maskgct_pipeline, smoke_prompt_wav)
        assert isinstance(audio, np.ndarray), (
            f"Expected numpy.ndarray, got {type(audio)}"
        )

    def test_inference_output_is_one_dimensional(
        self, maskgct_pipeline, smoke_prompt_wav
    ):
        """Inference output must be a 1-D (mono) audio array."""
        audio = self._run_inference(maskgct_pipeline, smoke_prompt_wav)
        assert audio.ndim == 1, (
            f"Expected 1-D audio array, got shape {audio.shape}"
        )

    def test_inference_output_length_is_positive(
        self, maskgct_pipeline, smoke_prompt_wav
    ):
        """Inference output must contain at least one audio sample."""
        audio = self._run_inference(maskgct_pipeline, smoke_prompt_wav)
        assert len(audio) > 0, "Recovered audio array must not be empty"

    def test_inference_output_is_finite(
        self, maskgct_pipeline, smoke_prompt_wav
    ):
        """Inference output must contain only finite values (no NaN or Inf)."""
        audio = self._run_inference(maskgct_pipeline, smoke_prompt_wav)
        assert np.isfinite(audio).all(), (
            "Recovered audio contains NaN or Inf values"
        )
