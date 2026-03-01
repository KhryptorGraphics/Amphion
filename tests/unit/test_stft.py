# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for utils/stft.py

Covers:
- window_sumsquare: output shape, dtype, non-negative values, win_length=None behaviour
- dynamic_range_compression: output shape, log semantics, clipping, compression factor C
- dynamic_range_decompression: inverse of exp, shape, C factor
- compress/decompress roundtrip: identity for values above clip_val
- STFT.inverse: output shape and finiteness (CPU, no CUDA required)
- STFT.transform: magnitude/phase shapes, non-negativity, phase range (CUDA)
- STFT.forward: end-to-end shape and reconstruction quality (CUDA)
- TacotronSTFT.mel_spectrogram: mel shape [B, n_mel, T], energy shape [B, T] (CUDA)
- griffin_lim: output shape and finiteness (CUDA)

Note: utils/stft.py uses the librosa 0.6 API for pad_center.  librosa >= 0.10
changed pad_center to a keyword-only ``size`` argument.  This module applies a
compatibility shim at import time so tests run on any installed librosa version
without touching the source file.

Note: STFT construction with large filter_length is slow (np.linalg.pinv on a
large matrix).  Tests use filter_length=128 to keep construction under ~1 s.
"""

import math

import librosa.filters as _librosa_filters
import librosa.util as _librosa_util
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Compatibility shims for librosa >= 0.10
#
# Two API changes affect utils/stft.py when running with librosa >= 0.10:
#
# 1. librosa.util.pad_center: second argument became keyword-only ``size``.
#    utils/stft.py calls it positionally: pad_center(data, n).
#
# 2. librosa.filters.mel: all arguments became keyword-only (*, sr, n_fft, …)
#    and ``n_mel_channels`` was renamed to ``n_mels``.
#    utils/stft.py calls it positionally:
#      librosa_mel_fn(sampling_rate, filter_length, n_mel_channels, fmin, fmax)
#
# We wrap both functions with positional-friendly signatures and patch them into
# the utils.stft namespace before any test runs.
# ---------------------------------------------------------------------------

_ORIG_PAD_CENTER = _librosa_util.pad_center
_ORIG_MEL = _librosa_filters.mel


def _compat_pad_center(data, size=None, axis=-1, **kwargs):
    """Accepts both positional pad_center(data, n) and keyword pad_center(data, size=n)."""
    return _ORIG_PAD_CENTER(data, size=size, axis=axis, **kwargs)


def _compat_librosa_mel(sr_or_kw=None, n_fft=None, n_mels=128, fmin=0.0, fmax=None, **kwargs):
    """Accepts the old positional API librosa_mel_fn(sr, n_fft, n_mels, fmin, fmax)."""
    return _ORIG_MEL(sr=sr_or_kw, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, **kwargs)


# Patch librosa.util attribute (fixes librosa_util.pad_center calls in window_sumsquare)
_librosa_util.pad_center = _compat_pad_center

# Now import utils.stft so its local name bindings are created, then replace them.
import utils.stft as _stft_module  # noqa: E402

_stft_module.pad_center = _compat_pad_center
_stft_module.librosa_mel_fn = _compat_librosa_mel

from utils.stft import (  # noqa: E402
    STFT,
    TacotronSTFT,
    dynamic_range_compression,
    dynamic_range_decompression,
    griffin_lim,
    window_sumsquare,
)

# ---------------------------------------------------------------------------
# CUDA availability guard
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="CUDA is not available on this machine"
)

# ---------------------------------------------------------------------------
# Module-level STFT parameters
#
# filter_length=128 keeps np.linalg.pinv inside STFT.__init__ under ~1 s.
# ---------------------------------------------------------------------------

_FL = 128  # filter_length
_HL = 32  # hop_length
_WL = 128  # win_length
_N_FREQ = _FL // 2 + 1  # 65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio(batch: int = 1, num_samples: int = 4096) -> torch.Tensor:
    """Return a float32 audio tensor of shape [batch, num_samples] in [-1, 1]."""
    torch.manual_seed(42)
    return torch.randn(batch, num_samples).clamp(-1.0, 1.0)


def _make_stft(filter_length=_FL, hop_length=_HL, win_length=_WL) -> STFT:
    return STFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
    )


def _make_tacotron_stft() -> TacotronSTFT:
    return TacotronSTFT(
        filter_length=_FL,
        hop_length=_HL,
        win_length=_WL,
        n_mel_channels=40,
        sampling_rate=16000,
        mel_fmin=0.0,
        mel_fmax=8000.0,
    )


# ---------------------------------------------------------------------------
# window_sumsquare
# ---------------------------------------------------------------------------


class TestWindowSumSquare:
    """Pure-NumPy helper – no CUDA or GPU required."""

    def test_output_shape(self):
        """Output length must equal n_fft + hop_length * (n_frames - 1)."""
        n_frames, hop_length, win_length, n_fft = 10, 32, 128, 128
        result = window_sumsquare("hann", n_frames, hop_length, win_length, n_fft)
        expected_len = n_fft + hop_length * (n_frames - 1)
        assert result.shape == (expected_len,)

    def test_output_dtype_float32(self):
        result = window_sumsquare("hann", 10, 32, 128, 128, dtype=np.float32)
        assert result.dtype == np.float32

    def test_output_dtype_float64(self):
        result = window_sumsquare("hann", 5, 32, 128, 128, dtype=np.float64)
        assert result.dtype == np.float64

    def test_output_nonnegative(self):
        """Sum of squared window values is always >= 0."""
        result = window_sumsquare("hann", 10, 32, 128, 128)
        assert np.all(result >= 0.0)

    def test_positive_in_covered_region(self):
        """At least some envelope values must be strictly positive."""
        result = window_sumsquare("hann", 10, 32, 128, 128)
        assert np.any(result > 0.0)

    def test_win_length_none_defaults_to_n_fft(self):
        """Passing win_length=None should be identical to win_length=n_fft."""
        n_fft = 128
        result_explicit = window_sumsquare("hann", 5, 32, n_fft, n_fft)
        result_none = window_sumsquare("hann", 5, 32, None, n_fft)
        np.testing.assert_array_equal(result_explicit, result_none)

    def test_single_frame(self):
        """With n_frames=1, output length equals n_fft."""
        n_fft = 64
        result = window_sumsquare("hann", 1, 32, 64, n_fft)
        assert result.shape == (n_fft,)

    def test_multiple_windows(self):
        """Rectangular and Hann windows both produce non-negative outputs."""
        for win in ("hann", "hamming", "boxcar"):
            result = window_sumsquare(win, 8, 32, 128, 128)
            assert np.all(result >= 0.0), f"Negative values for window '{win}'"


# ---------------------------------------------------------------------------
# dynamic_range_compression
# ---------------------------------------------------------------------------


class TestDynamicRangeCompression:
    def test_output_shape_preserved(self):
        x = torch.ones(4, 80, 100)
        out = dynamic_range_compression(x)
        assert out.shape == x.shape

    def test_log_compression_values(self):
        """log(C * x) with C=1 and x > clip_val equals torch.log(x)."""
        x = torch.tensor([1.0, math.e, math.e**2])
        out = dynamic_range_compression(x, C=1, clip_val=1e-10)
        expected = torch.log(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_clip_prevents_negative_infinity(self):
        """Values at or below zero are clamped so output remains finite."""
        x = torch.tensor([0.0, -1.0, 1e-10])
        out = dynamic_range_compression(x, C=1, clip_val=1e-5)
        assert torch.all(torch.isfinite(out))

    def test_clip_val_applied_correctly(self):
        """Values below clip_val are clamped to clip_val before log."""
        clip_val = 1e-5
        x = torch.tensor([0.0, 1e-10, clip_val / 2])
        out = dynamic_range_compression(x, C=1, clip_val=clip_val)
        expected_val = math.log(clip_val)
        assert torch.allclose(out, torch.full_like(out, expected_val), atol=1e-4)

    def test_compression_factor_C(self):
        """C multiplier scales before log: output = log(C * x)."""
        x = torch.tensor([1.0])
        C = 5.0
        out = dynamic_range_compression(x, C=C, clip_val=1e-10)
        expected = math.log(C)
        assert abs(out.item() - expected) < 1e-5

    def test_output_finite_for_typical_input(self):
        x = torch.rand(16, 80, 200) + 1e-4
        out = dynamic_range_compression(x)
        assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# dynamic_range_decompression
# ---------------------------------------------------------------------------


class TestDynamicRangeDecompression:
    def test_output_shape_preserved(self):
        x = torch.zeros(4, 80, 100)
        out = dynamic_range_decompression(x)
        assert out.shape == x.shape

    def test_exp_inverse(self):
        """With C=1, decompression is just exp(x)."""
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        out = dynamic_range_decompression(x, C=1)
        expected = torch.exp(x)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_compression_factor_C_divides(self):
        """Result equals exp(x) / C."""
        C = 3.0
        x = torch.tensor([0.0])
        out = dynamic_range_decompression(x, C=C)
        expected = math.exp(0.0) / C
        assert abs(out.item() - expected) < 1e-6

    def test_output_positive_for_real_inputs(self):
        """exp of any real number is strictly positive."""
        x = torch.randn(4, 80, 50)
        out = dynamic_range_decompression(x)
        assert torch.all(out > 0.0)


# ---------------------------------------------------------------------------
# Compression / decompression roundtrip
# ---------------------------------------------------------------------------


class TestCompressionDecompressionRoundtrip:
    def test_roundtrip_identity(self):
        """compress then decompress recovers the original for x >> clip_val."""
        x = torch.rand(4, 80, 100) * 0.9 + 0.01  # well above clip_val=1e-5
        compressed = dynamic_range_compression(x, C=1, clip_val=1e-5)
        recovered = dynamic_range_decompression(compressed, C=1)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_roundtrip_with_nonunit_C(self):
        """Roundtrip should hold for any positive C."""
        C = 4.0
        x = torch.rand(2, 32) * 0.9 + 0.01
        compressed = dynamic_range_compression(x, C=C, clip_val=1e-5)
        recovered = dynamic_range_decompression(compressed, C=C)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_roundtrip_scalar(self):
        C = 2.5
        x = torch.tensor([0.5])
        out = dynamic_range_decompression(dynamic_range_compression(x, C=C), C=C)
        assert torch.allclose(x, out, atol=1e-5)


# ---------------------------------------------------------------------------
# STFT.inverse  (CPU, no CUDA needed)
# ---------------------------------------------------------------------------


class TestSTFTInverse:
    """Tests for STFT.inverse that run entirely on CPU.

    STFT.inverse does not call .cuda(); it only calls window_sumsquare
    internally, which calls librosa_util.pad_center – patched at module level.
    """

    @pytest.fixture(autouse=True)
    def stft(self):
        self._stft = _make_stft()

    def test_inverse_output_ndim(self):
        magnitude = torch.rand(2, _N_FREQ, 32)
        phase = torch.zeros(2, _N_FREQ, 32)
        out = self._stft.inverse(magnitude, phase)
        assert out.ndim == 3

    def test_inverse_batch_dim_preserved(self):
        batch = 3
        magnitude = torch.rand(batch, _N_FREQ, 20)
        phase = torch.zeros(batch, _N_FREQ, 20)
        out = self._stft.inverse(magnitude, phase)
        assert out.shape[0] == batch

    def test_inverse_channel_dim_is_one(self):
        """inverse returns [batch, 1, T] – channel is always 1."""
        magnitude = torch.rand(2, _N_FREQ, 20)
        phase = torch.zeros(2, _N_FREQ, 20)
        out = self._stft.inverse(magnitude, phase)
        assert out.shape[1] == 1

    def test_inverse_time_dim_positive(self):
        magnitude = torch.rand(1, _N_FREQ, 10)
        phase = torch.zeros(1, _N_FREQ, 10)
        out = self._stft.inverse(magnitude, phase)
        assert out.shape[2] > 0

    def test_inverse_output_finite(self):
        magnitude = torch.rand(2, _N_FREQ, 32)
        phase = torch.rand(2, _N_FREQ, 32) * math.pi
        out = self._stft.inverse(magnitude, phase)
        assert torch.all(torch.isfinite(out))

    def test_inverse_zero_magnitude_gives_silence(self):
        """Zero magnitude should produce a near-zero output signal."""
        magnitude = torch.zeros(1, _N_FREQ, 16)
        phase = torch.zeros(1, _N_FREQ, 16)
        out = self._stft.inverse(magnitude, phase)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# ---------------------------------------------------------------------------
# STFT.transform  (requires CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
class TestSTFTTransform:
    """Tests for STFT.transform – hard-codes .cuda() so CUDA is required."""

    @pytest.fixture(autouse=True)
    def stft(self):
        self._stft = _make_stft()

    def test_transform_magnitude_shape(self):
        """Magnitude: [batch, filter_length//2+1, T_frames]."""
        x = _make_audio(batch=2, num_samples=4096)
        magnitude, phase = self._stft.transform(x)
        assert magnitude.ndim == 3
        assert magnitude.shape[0] == 2
        assert magnitude.shape[1] == _N_FREQ
        assert magnitude.shape[2] > 0

    def test_transform_phase_shape_matches_magnitude(self):
        x = _make_audio(batch=2, num_samples=4096)
        magnitude, phase = self._stft.transform(x)
        assert phase.shape == magnitude.shape

    def test_transform_magnitude_nonnegative(self):
        """Magnitude = sqrt(real^2 + imag^2) >= 0."""
        x = _make_audio(batch=1, num_samples=4096)
        magnitude, _ = self._stft.transform(x)
        assert torch.all(magnitude >= 0.0)

    def test_transform_phase_in_valid_range(self):
        """Phase is atan2 output, so lies in [-pi, pi]."""
        x = _make_audio(batch=1, num_samples=4096)
        _, phase = self._stft.transform(x)
        assert torch.all(phase >= -math.pi - 1e-5)
        assert torch.all(phase <= math.pi + 1e-5)

    def test_transform_output_finite(self):
        x = _make_audio(batch=1, num_samples=4096)
        magnitude, phase = self._stft.transform(x)
        assert torch.all(torch.isfinite(magnitude))
        assert torch.all(torch.isfinite(phase))

    def test_transform_batch_independence(self):
        """Two identical rows in a batch produce identical magnitude outputs."""
        x = _make_audio(batch=1, num_samples=4096)
        x2 = torch.cat([x, x], dim=0)  # batch=2, both rows identical
        mag2, _ = self._stft.transform(x2)
        assert torch.allclose(mag2[0], mag2[1], atol=1e-5)


# ---------------------------------------------------------------------------
# STFT.forward  (requires CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
class TestSTFTForward:
    """Tests for STFT.forward (transform + inverse end-to-end)."""

    @pytest.fixture(autouse=True)
    def stft(self):
        self._stft = _make_stft()

    def test_forward_output_ndim(self):
        x = _make_audio(batch=2, num_samples=4096)
        out = self._stft.forward(x)
        assert out.ndim == 3

    def test_forward_batch_dim_preserved(self):
        batch = 2
        x = _make_audio(batch=batch, num_samples=4096)
        out = self._stft.forward(x)
        assert out.shape[0] == batch

    def test_forward_channel_dim_is_one(self):
        """STFT.forward returns [batch, 1, T] – channel always 1."""
        x = _make_audio(batch=1, num_samples=4096)
        out = self._stft.forward(x)
        assert out.shape[1] == 1

    def test_forward_time_dim_positive(self):
        x = _make_audio(batch=1, num_samples=4096)
        out = self._stft.forward(x)
        assert out.shape[2] > 0

    def test_forward_reconstruction_finite(self):
        x = _make_audio(batch=1, num_samples=4096)
        out = self._stft.forward(x)
        assert torch.all(torch.isfinite(out))

    def test_forward_reconstruction_quality(self):
        """Round-trip STFT/ISTFT must correlate well with the original signal."""
        torch.manual_seed(0)
        x = torch.randn(1, 8192).clamp(-1.0, 1.0)
        reconstruction = self._stft.forward(x).squeeze()  # [T_rec]
        min_len = min(x.shape[1], reconstruction.shape[0])
        orig = x[0, :min_len]
        rec = reconstruction[:min_len]
        corr = torch.dot(orig, rec) / (torch.norm(orig) * torch.norm(rec) + 1e-8)
        assert corr.item() > 0.5, (
            f"Reconstruction correlation too low: {corr.item():.3f}"
        )


# ---------------------------------------------------------------------------
# TacotronSTFT.mel_spectrogram  (requires CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
class TestTacotronSTFT:
    """Tests for TacotronSTFT.mel_spectrogram."""

    @pytest.fixture(autouse=True)
    def tac_stft(self):
        self._tac = _make_tacotron_stft()

    def test_mel_output_ndim(self):
        y = _make_audio(batch=2, num_samples=4096)
        mel_output, _ = self._tac.mel_spectrogram(y)
        assert mel_output.ndim == 3

    def test_mel_output_batch_dim(self):
        batch = 2
        y = _make_audio(batch=batch, num_samples=4096)
        mel_output, _ = self._tac.mel_spectrogram(y)
        assert mel_output.shape[0] == batch

    def test_mel_output_n_mel_dim(self):
        """Second dimension must equal n_mel_channels=40."""
        y = _make_audio(batch=1, num_samples=4096)
        mel_output, _ = self._tac.mel_spectrogram(y)
        assert mel_output.shape[1] == 40

    def test_mel_output_time_dim_positive(self):
        y = _make_audio(batch=1, num_samples=4096)
        mel_output, _ = self._tac.mel_spectrogram(y)
        assert mel_output.shape[2] > 0

    def test_energy_ndim(self):
        """Energy tensor must be 2-D: [batch, T_frames]."""
        y = _make_audio(batch=2, num_samples=4096)
        _, energy = self._tac.mel_spectrogram(y)
        assert energy.ndim == 2

    def test_energy_batch_dim(self):
        batch = 2
        y = _make_audio(batch=batch, num_samples=4096)
        _, energy = self._tac.mel_spectrogram(y)
        assert energy.shape[0] == batch

    def test_energy_time_dim_matches_mel(self):
        """Energy and mel-spectrogram must share the time dimension."""
        y = _make_audio(batch=2, num_samples=4096)
        mel_output, energy = self._tac.mel_spectrogram(y)
        assert energy.shape[1] == mel_output.shape[2]

    def test_mel_output_finite(self):
        y = _make_audio(batch=1, num_samples=4096)
        mel_output, energy = self._tac.mel_spectrogram(y)
        assert torch.all(torch.isfinite(mel_output))
        assert torch.all(torch.isfinite(energy))

    def test_energy_nonnegative(self):
        """Energy = torch.norm of magnitudes, always >= 0."""
        y = _make_audio(batch=1, num_samples=4096)
        _, energy = self._tac.mel_spectrogram(y)
        assert torch.all(energy >= 0.0)

    def test_mel_log_compressed(self):
        """Mel output is log-compressed; most values should be negative."""
        y = _make_audio(batch=1, num_samples=4096)
        mel_output, _ = self._tac.mel_spectrogram(y)
        fraction_negative = torch.mean((mel_output < 0).float()).item()
        assert fraction_negative > 0.3, (
            f"Expected mostly negative log-mel values, "
            f"got {fraction_negative:.2f} fraction negative"
        )

    def test_input_out_of_range_raises(self):
        """mel_spectrogram asserts input in [-1, 1]; values outside raise AssertionError."""
        y = torch.ones(1, 4096) * 1.5  # exceeds max=1
        with pytest.raises(AssertionError):
            self._tac.mel_spectrogram(y)


# ---------------------------------------------------------------------------
# griffin_lim  (requires CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
class TestGriffinLim:
    """Tests for griffin_lim phase reconstruction."""

    @pytest.fixture(autouse=True)
    def stft(self):
        self._stft = _make_stft()

    def test_output_ndim(self):
        magnitudes = torch.rand(1, _N_FREQ, 32)
        signal = griffin_lim(magnitudes, self._stft, n_iters=2)
        assert signal.ndim == 2

    def test_output_batch_dim(self):
        batch = 1
        magnitudes = torch.rand(batch, _N_FREQ, 32)
        signal = griffin_lim(magnitudes, self._stft, n_iters=2)
        assert signal.shape[0] == batch

    def test_output_time_dim_positive(self):
        magnitudes = torch.rand(1, _N_FREQ, 32)
        signal = griffin_lim(magnitudes, self._stft, n_iters=2)
        assert signal.shape[1] > 0

    def test_output_finite(self):
        magnitudes = torch.rand(1, _N_FREQ, 20)
        signal = griffin_lim(magnitudes, self._stft, n_iters=2)
        assert torch.all(torch.isfinite(signal))

    def test_single_iteration(self):
        """Verify griffin_lim completes successfully with n_iters=1."""
        magnitudes = torch.rand(1, _N_FREQ, 16)
        signal = griffin_lim(magnitudes, self._stft, n_iters=1)
        assert signal.shape[0] == 1
        assert torch.all(torch.isfinite(signal))
