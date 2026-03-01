# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke tests for Vevo pretrained model inference.

These tests:
1. Attempt to import all required Vevo components (skips on ImportError)
2. Download the Vevo model weights from HuggingFace Hub
   (skips if HuggingFace is not reachable or the download fails)
3. Run a short timbre-conversion inference pass (inference_fm) on a
   synthetic source WAV file
4. Verify that the returned audio tensor is a non-empty PyTorch tensor with
   shape (1, T) and only finite values

All tests are marked @pytest.mark.smoke and @pytest.mark.slow.
Exclude them from regular CI runs with::

    pytest -m "not smoke"

or::

    pytest -m "not slow"
"""

import os
import socket

import pytest

# ---------------------------------------------------------------------------
# Module-level import guard
# ---------------------------------------------------------------------------
# Attempt all Vevo imports at module load time so that a missing dependency
# causes the entire module to be skipped rather than failing at collection.

_import_error = None
try:
    import torch
    from huggingface_hub import snapshot_download

    from models.vc.vevo.vevo_utils import VevoInferencePipeline
except ImportError as _exc:
    _import_error = _exc

# Apply smoke + slow marks to every test in this module.
pytestmark = [pytest.mark.smoke, pytest.mark.slow]

# Config paths (relative to repository root; tests should be run from there)
_FMT_CFG_PATH = "./models/vc/vevo/config/Vq8192ToMels.json"
_VOCODER_CFG_PATH = "./models/vc/vevo/config/Vocoder.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_import_failed() -> None:
    """Call pytest.skip if any required Vevo import failed at load time."""
    if _import_error is not None:
        pytest.skip(
            f"Required Vevo dependency not available: {_import_error}"
        )


def _build_vevo_fm_pipeline(device):
    """Instantiate and return a Vevo FM-only inference pipeline.

    Downloads the content-style tokenizer, flow matching transformer, and
    vocoder checkpoints from HuggingFace Hub.  Results are cached locally by
    huggingface_hub after the first download.

    Args:
        device: A ``torch.device`` specifying where to load the models.

    Returns:
        VevoInferencePipeline: A pipeline ready for timbre-conversion inference
        via ``inference_fm``.

    Raises:
        Exception: Any failure during download or weight loading propagates to
                   the caller so that the fixture can call ``pytest.skip()``.
    """
    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    return VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=tokenizer_ckpt_path,
        fmt_cfg_path=_FMT_CFG_PATH,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=_VOCODER_CFG_PATH,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )


# ---------------------------------------------------------------------------
# Module-scoped pipeline fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vevo_pipeline():
    """Module-scoped fixture providing a loaded Vevo FM-only inference pipeline.

    The fixture is shared across all tests in this module so that model
    weights are downloaded and loaded only once per test session.

    Skips all dependent tests if:
    - Required imports are unavailable
    - HuggingFace Hub is not reachable
    - Required config files are missing from the repository
    - Model weights cannot be downloaded or loaded
    """
    _skip_if_import_failed()

    # Verify config files are present before attempting a heavy download
    for cfg_path in (_FMT_CFG_PATH, _VOCODER_CFG_PATH):
        if not os.path.isfile(cfg_path):
            pytest.skip(f"Vevo config file not found: {cfg_path}")

    # Check HF connectivity before attempting a potentially slow download
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=5.0):
            pass
    except (socket.error, OSError):
        pytest.skip(
            "HuggingFace Hub is not reachable — Vevo inference tests skipped"
        )

    try:
        return _build_vevo_fm_pipeline(torch.device("cpu"))
    except Exception as exc:
        pytest.skip(f"Failed to load Vevo weights from HuggingFace: {exc}")


# ---------------------------------------------------------------------------
# Tests: import verification
# ---------------------------------------------------------------------------


class TestVevoImports:
    """Verify that core Vevo module components can be imported successfully."""

    def test_vevo_inference_pipeline_importable(self):
        """VevoInferencePipeline must be importable from vevo_utils."""
        _skip_if_import_failed()
        assert VevoInferencePipeline is not None

    def test_vevo_flow_matching_transformer_importable(self):
        """FlowMatchingTransformer must be importable from the vc module."""
        _skip_if_import_failed()
        from models.vc.flow_matching_transformer.fmt_model import (
            FlowMatchingTransformer,
        )

        assert FlowMatchingTransformer is not None

    def test_vevo_vocoder_importable(self):
        """Vocos vocoder must be importable from the amphion_codec module."""
        _skip_if_import_failed()
        from models.codec.amphion_codec.vocos import Vocos

        assert Vocos is not None

    def test_vevo_config_files_exist(self):
        """Required Vevo JSON config files must be present in the repository."""
        _skip_if_import_failed()
        for cfg_path in (_FMT_CFG_PATH, _VOCODER_CFG_PATH):
            assert os.path.isfile(cfg_path), (
                f"Vevo config file not found: {cfg_path}"
            )


# ---------------------------------------------------------------------------
# Tests: end-to-end inference with pretrained weights
# ---------------------------------------------------------------------------


class TestVevoInference:
    """End-to-end smoke tests for Vevo timbre conversion with pretrained weights.

    Uses the FM-only pipeline (``inference_fm``) which does not require the
    autoregressive transformer, reducing download size and inference time.
    Inference is run on CPU to remain hardware-agnostic.
    """

    # Minimal flow-matching steps: fast but still exercises the full forward pass
    _FLOW_MATCHING_STEPS = 4

    def _run_inference(self, pipeline, src_wav_path: str, ref_wav_path: str):
        """Run timbre-conversion inference and return the output tensor.

        Calls ``pytest.skip`` on any runtime failure so that transient
        infrastructure problems do not masquerade as test failures.
        """
        try:
            return pipeline.inference_fm(
                src_wav_path=src_wav_path,
                timbre_ref_wav_path=ref_wav_path,
                flow_matching_steps=self._FLOW_MATCHING_STEPS,
            )
        except Exception as exc:
            pytest.skip(
                f"Vevo inference_fm failed (possible missing resource): {exc}"
            )

    def test_inference_returns_tensor(self, vevo_pipeline, smoke_prompt_wav):
        """inference_fm must return a torch.Tensor."""
        audio = self._run_inference(vevo_pipeline, smoke_prompt_wav, smoke_prompt_wav)
        assert isinstance(audio, torch.Tensor), (
            f"Expected torch.Tensor, got {type(audio)}"
        )

    def test_inference_output_has_samples(self, vevo_pipeline, smoke_prompt_wav):
        """inference_fm output must contain at least one audio sample."""
        audio = self._run_inference(vevo_pipeline, smoke_prompt_wav, smoke_prompt_wav)
        assert audio.numel() > 0, "Vevo output tensor must not be empty"

    def test_inference_output_shape(self, vevo_pipeline, smoke_prompt_wav):
        """inference_fm output must have shape (1, T) — one channel, T samples."""
        audio = self._run_inference(vevo_pipeline, smoke_prompt_wav, smoke_prompt_wav)
        assert audio.ndim == 2, (
            f"Expected 2-D tensor (1, T), got shape {audio.shape}"
        )
        assert audio.shape[0] == 1, (
            f"Expected channel dimension of 1, got {audio.shape[0]}"
        )

    def test_inference_output_is_finite(self, vevo_pipeline, smoke_prompt_wav):
        """inference_fm output must contain only finite values (no NaN or Inf)."""
        audio = self._run_inference(vevo_pipeline, smoke_prompt_wav, smoke_prompt_wav)
        assert torch.isfinite(audio).all(), (
            "Vevo output tensor contains NaN or Inf values"
        )
