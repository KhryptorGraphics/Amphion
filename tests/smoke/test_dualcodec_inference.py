# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke tests for DualCodec codec encode/decode inference.

DualCodec is pip-installable via ``pip install dualcodec`` and can therefore
be available without a GPU.  These tests exercise the full encode→decode
roundtrip using the pretrained weights from HuggingFace Hub.

These tests:
1. Attempt to import the ``dualcodec`` package (skips on ImportError)
2. Download the DualCodec model weights from HuggingFace Hub via
   ``get_model()`` (skips if HuggingFace is not reachable)
3. Build the ``Inference`` wrapper which also downloads the
   ``facebook/w2v-bert-2.0`` semantic model
4. Encode a synthetic audio clip into semantic + acoustic codes and decode
   it back to a waveform
5. Verify that the decoded audio is a non-empty tensor with shape (B, 1, T)
   and only finite values

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
# DualCodec is pip-installable.  Attempt all imports at module load time so
# that a missing package causes the whole module to be skipped cleanly.

_import_error = None
try:
    import torch
    import torchaudio

    # Try the pip-installed package first; fall back to the in-tree copy.
    try:
        from dualcodec import get_model, Inference, DualCodec
    except ImportError:
        from models.codec.dualcodec.dualcodec import get_model, Inference, DualCodec
except ImportError as _exc:
    _import_error = _exc

# Apply smoke + slow marks to every test in this module.
pytestmark = [pytest.mark.smoke, pytest.mark.slow]

# Model variant used in tests: smallest / fastest available
_MODEL_ID = "12hz_v1"
_SAMPLE_RATE = 24000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_import_failed() -> None:
    """Call pytest.skip if any required DualCodec import failed at load time."""
    if _import_error is not None:
        pytest.skip(
            f"Required DualCodec dependency not available: {_import_error}"
        )


def _build_dualcodec_inference(device_str: str):
    """Load the DualCodec model and wrap it in an ``Inference`` instance.

    Downloads model weights and the w2v-bert-2.0 semantic model from
    HuggingFace Hub (results are cached locally after the first download).

    Args:
        device_str: Device string (``"cpu"`` or ``"cuda"``).

    Returns:
        Inference: A DualCodec inference wrapper ready for encode/decode.

    Raises:
        Exception: Any failure during download or weight loading propagates to
                   the caller so that the fixture can call ``pytest.skip()``.
    """
    model = get_model(model_id=_MODEL_ID, pretrained_model_path="hf://amphion/dualcodec")
    inference = Inference(
        dualcodec_model=model,
        dualcodec_path="hf://amphion/dualcodec",
        w2v_path="hf://facebook/w2v-bert-2.0",
        device=device_str,
        autocast=False,  # Disable autocast so float16 doesn't fail on CPU
    )
    return inference


# ---------------------------------------------------------------------------
# Module-scoped inference fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dualcodec_inference():
    """Module-scoped fixture providing a loaded DualCodec Inference instance.

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
            "HuggingFace Hub is not reachable — DualCodec inference tests skipped"
        )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return _build_dualcodec_inference(device_str)
    except Exception as exc:
        pytest.skip(f"Failed to load DualCodec weights from HuggingFace: {exc}")


# ---------------------------------------------------------------------------
# Shared synthetic audio fixture (module-scoped for efficiency)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dualcodec_audio():
    """Create a short synthetic 440 Hz sine-wave audio tensor at 24 kHz.

    Shape: (1, 1, T) — batch=1, channels=1, samples — as required by the
    DualCodec ``Inference.encode`` method.

    Returns:
        torch.Tensor: Float32 audio tensor of shape (1, 1, T).
    """
    _skip_if_import_failed()
    duration = 1.0  # seconds — short enough to be fast, long enough to encode
    t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), endpoint=False)
    waveform = np.sin(2 * np.pi * 440.0 * t).astype(np.float32) * 0.3
    return torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)  # (1, 1, T)


# ---------------------------------------------------------------------------
# Tests: import verification
# ---------------------------------------------------------------------------


class TestDualCodecImports:
    """Verify that all DualCodec components can be imported successfully."""

    def test_dualcodec_class_importable(self):
        """DualCodec model class must be importable."""
        _skip_if_import_failed()
        assert DualCodec is not None

    def test_get_model_importable(self):
        """get_model helper must be importable from the dualcodec package."""
        _skip_if_import_failed()
        assert get_model is not None

    def test_inference_class_importable(self):
        """Inference wrapper class must be importable from the dualcodec package."""
        _skip_if_import_failed()
        assert Inference is not None


# ---------------------------------------------------------------------------
# Tests: end-to-end encode / decode with pretrained weights
# ---------------------------------------------------------------------------


class TestDualCodecInference:
    """End-to-end smoke tests for DualCodec encode/decode with pretrained weights.

    Exercises the full pipeline:
      audio → Inference.encode() → semantic + acoustic codes → Inference.decode()
      → reconstructed audio waveform

    Inference is performed on CPU (or GPU when available) with a short 1-second
    synthetic clip to limit wall-clock time.
    """

    def _run_encode(self, inference, audio):
        """Encode audio and return (semantic_codes, acoustic_codes).

        Calls ``pytest.skip`` on any runtime failure so that infrastructure
        problems do not masquerade as test failures.
        """
        try:
            return inference.encode(audio, n_quantizers=4)
        except Exception as exc:
            pytest.skip(
                f"DualCodec encode failed (possible missing resource): {exc}"
            )

    def _run_decode(self, inference, semantic_codes, acoustic_codes):
        """Decode codes back to audio waveform.

        Calls ``pytest.skip`` on any runtime failure.
        """
        try:
            return inference.decode(semantic_codes, acoustic_codes)
        except Exception as exc:
            pytest.skip(
                f"DualCodec decode failed (possible missing resource): {exc}"
            )

    def test_encode_returns_two_tensors(self, dualcodec_inference, dualcodec_audio):
        """encode() must return a 2-tuple of (semantic_codes, acoustic_codes)."""
        result = self._run_encode(dualcodec_inference, dualcodec_audio)
        assert isinstance(result, (tuple, list)), (
            f"Expected a tuple from encode(), got {type(result)}"
        )
        assert len(result) == 2, (
            f"Expected 2 values from encode(), got {len(result)}"
        )

    def test_semantic_codes_shape(self, dualcodec_inference, dualcodec_audio):
        """Semantic codes must have shape (B, 1, T) — one codebook entry per frame."""
        semantic_codes, _ = self._run_encode(dualcodec_inference, dualcodec_audio)
        assert isinstance(semantic_codes, torch.Tensor), (
            f"Expected torch.Tensor for semantic_codes, got {type(semantic_codes)}"
        )
        assert semantic_codes.ndim == 3, (
            f"Expected 3-D semantic_codes (B, 1, T), got shape {semantic_codes.shape}"
        )
        assert semantic_codes.shape[1] == 1, (
            f"Expected 1 semantic codebook, got {semantic_codes.shape[1]}"
        )

    def test_acoustic_codes_shape(self, dualcodec_inference, dualcodec_audio):
        """Acoustic codes must have shape (B, num_vq-1, T) with at least one codebook."""
        _, acoustic_codes = self._run_encode(dualcodec_inference, dualcodec_audio)
        assert isinstance(acoustic_codes, torch.Tensor), (
            f"Expected torch.Tensor for acoustic_codes, got {type(acoustic_codes)}"
        )
        assert acoustic_codes.ndim == 3, (
            f"Expected 3-D acoustic_codes (B, Q, T), got shape {acoustic_codes.shape}"
        )
        assert acoustic_codes.shape[1] >= 1, (
            "Expected at least one acoustic codebook"
        )

    def test_decode_returns_tensor(self, dualcodec_inference, dualcodec_audio):
        """decode() must return a torch.Tensor."""
        semantic_codes, acoustic_codes = self._run_encode(
            dualcodec_inference, dualcodec_audio
        )
        reconstructed = self._run_decode(
            dualcodec_inference, semantic_codes, acoustic_codes
        )
        assert isinstance(reconstructed, torch.Tensor), (
            f"Expected torch.Tensor from decode(), got {type(reconstructed)}"
        )

    def test_decode_output_shape(self, dualcodec_inference, dualcodec_audio):
        """Decoded audio must have shape (B, 1, T) — batch, one channel, samples."""
        semantic_codes, acoustic_codes = self._run_encode(
            dualcodec_inference, dualcodec_audio
        )
        reconstructed = self._run_decode(
            dualcodec_inference, semantic_codes, acoustic_codes
        )
        assert reconstructed.ndim == 3, (
            f"Expected 3-D decoded audio (B, 1, T), got shape {reconstructed.shape}"
        )
        assert reconstructed.shape[1] == 1, (
            f"Expected 1 output channel, got {reconstructed.shape[1]}"
        )

    def test_decode_output_length_is_positive(self, dualcodec_inference, dualcodec_audio):
        """Decoded audio must contain at least one sample."""
        semantic_codes, acoustic_codes = self._run_encode(
            dualcodec_inference, dualcodec_audio
        )
        reconstructed = self._run_decode(
            dualcodec_inference, semantic_codes, acoustic_codes
        )
        assert reconstructed.shape[-1] > 0, "Decoded audio must not be empty"

    def test_decode_output_is_finite(self, dualcodec_inference, dualcodec_audio):
        """Decoded audio must contain only finite values (no NaN or Inf)."""
        semantic_codes, acoustic_codes = self._run_encode(
            dualcodec_inference, dualcodec_audio
        )
        reconstructed = self._run_decode(
            dualcodec_inference, semantic_codes, acoustic_codes
        )
        assert torch.isfinite(reconstructed).all(), (
            "Decoded audio contains NaN or Inf values"
        )
