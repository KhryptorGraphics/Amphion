# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for evaluation/metrics/spectrogram/ modules.

Covers:
- extract_si_sdr: SI-SDR of identical signals should be high (near-infinite)
- extract_si_snr: SI-SNR of identical signals should be high (near-infinite)
- extract_mstft: Multi-resolution STFT distance for identical audio should be 0

All tests use temporary WAV files created with soundfile.
Tests are skipped if torchmetrics is not available (required by SI-SDR/SI-SNR).
"""

import numpy as np
import pytest
import soundfile as sf

# Skip the entire module if torch is not installed — all metrics require it.
pytest.importorskip("torch", reason="torch not installed; skipping spectrogram evaluation tests")

# Skip if torchmetrics is not installed — SI-SDR and SI-SNR depend on it.
torchmetrics = pytest.importorskip(
    "torchmetrics",
    reason="torchmetrics not installed; skipping spectrogram evaluation tests",
)

# ---------------------------------------------------------------------------
# Audio creation helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_DURATION = 1.0  # seconds


def _make_sine_wav(tmp_path, filename, freq=200.0, duration=_DURATION, sample_rate=_SAMPLE_RATE):
    """Create a temporary WAV file containing a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / filename
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


def _make_noise_wav(tmp_path, filename, duration=_DURATION, sample_rate=_SAMPLE_RATE, seed=42):
    """Create a temporary WAV file containing Gaussian white noise."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(sample_rate * duration)) * 0.1).astype(np.float32)
    wav_path = tmp_path / filename
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


# ---------------------------------------------------------------------------
# extract_si_sdr
# ---------------------------------------------------------------------------


class TestExtractSiSdr:
    """Tests for evaluation/metrics/spectrogram/scale_invariant_signal_to_distortion_ratio.py."""

    def test_identical_audio_returns_high_si_sdr(self, tmp_path):
        """Identical audio files should produce a very high SI-SDR value.

        When the degraded signal equals the reference signal exactly, the
        distortion term is zero and SI-SDR approaches infinity. In practice
        torchmetrics clamps or returns a very large finite value; we check
        that it exceeds a generous lower bound.
        """
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import (
            extract_si_sdr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_si_sdr(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"SI-SDR must return a numeric value, got {type(result)}"
        )
        assert result > 20.0, (
            f"SI-SDR for identical audio should be high (> 20 dB), got {result}"
        )

    def test_returns_numeric_value(self, tmp_path):
        """extract_si_sdr must return a single numeric (int or float) value."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import (
            extract_si_sdr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_si_sdr(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"extract_si_sdr must return a numeric value, got {type(result)}"
        )

    def test_noisy_audio_lower_si_sdr_than_identical(self, tmp_path):
        """Noisy degraded audio should produce a lower SI-SDR than the identical case."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import (
            extract_si_sdr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_identical = _make_sine_wav(tmp_path, "deg_identical.wav")
        deg_noisy = _make_noise_wav(tmp_path, "deg_noisy.wav")

        si_sdr_identical = extract_si_sdr(
            ref_path,
            deg_identical,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )
        si_sdr_noisy = extract_si_sdr(
            ref_path,
            deg_noisy,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert si_sdr_identical > si_sdr_noisy, (
            f"SI-SDR for identical audio ({si_sdr_identical:.2f}) should exceed "
            f"SI-SDR for noisy audio ({si_sdr_noisy:.2f})"
        )

    def test_cut_method_identical_audio(self, tmp_path):
        """Method='cut' should work correctly and return a high SI-SDR for identical audio."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import (
            extract_si_sdr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav", duration=1.0)
        deg_path = _make_sine_wav(tmp_path, "deg.wav", duration=1.0)

        result = extract_si_sdr(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert result > 20.0, (
            f"SI-SDR (cut) for identical audio should be high, got {result}"
        )

    def test_no_fs_uses_native_sample_rate(self, tmp_path):
        """Passing fs=None should load audio at its native sample rate without error."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_distortion_ratio import (
            extract_si_sdr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_si_sdr(
            ref_path,
            deg_path,
            kwargs={"fs": None, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"extract_si_sdr with fs=None must return a numeric value, got {type(result)}"
        )


# ---------------------------------------------------------------------------
# extract_si_snr
# ---------------------------------------------------------------------------


class TestExtractSiSnr:
    """Tests for evaluation/metrics/spectrogram/scale_invariant_signal_to_noise_ratio.py."""

    def test_identical_audio_returns_high_si_snr(self, tmp_path):
        """Identical audio files should produce a very high SI-SNR value.

        When the degraded signal equals the reference, the noise term is zero
        and SI-SNR approaches infinity. We check for a generous lower bound.
        """
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_noise_ratio import (
            extract_si_snr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_si_snr(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"SI-SNR must return a numeric value, got {type(result)}"
        )
        assert result > 20.0, (
            f"SI-SNR for identical audio should be high (> 20 dB), got {result}"
        )

    def test_returns_numeric_value(self, tmp_path):
        """extract_si_snr must return a single numeric (int or float) value."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_noise_ratio import (
            extract_si_snr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_si_snr(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"extract_si_snr must return a numeric value, got {type(result)}"
        )

    def test_noisy_audio_lower_si_snr_than_identical(self, tmp_path):
        """Noisy degraded audio should produce a lower SI-SNR than the identical case."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_noise_ratio import (
            extract_si_snr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_identical = _make_sine_wav(tmp_path, "deg_identical.wav")
        deg_noisy = _make_noise_wav(tmp_path, "deg_noisy.wav")

        si_snr_identical = extract_si_snr(
            ref_path,
            deg_identical,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )
        si_snr_noisy = extract_si_snr(
            ref_path,
            deg_noisy,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert si_snr_identical > si_snr_noisy, (
            f"SI-SNR for identical audio ({si_snr_identical:.2f}) should exceed "
            f"SI-SNR for noisy audio ({si_snr_noisy:.2f})"
        )

    def test_no_fs_uses_native_sample_rate(self, tmp_path):
        """Passing fs=None should load audio at its native sample rate without error."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_noise_ratio import (
            extract_si_snr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_si_snr(
            ref_path,
            deg_path,
            kwargs={"fs": None, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"extract_si_snr with fs=None must return a numeric value, got {type(result)}"
        )

    def test_cut_method_identical_audio(self, tmp_path):
        """Method='cut' should work correctly and return a high SI-SNR for identical audio."""
        from evaluation.metrics.spectrogram.scale_invariant_signal_to_noise_ratio import (
            extract_si_snr,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_si_snr(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert result > 20.0, (
            f"SI-SNR (cut) for identical audio should be high, got {result}"
        )


# ---------------------------------------------------------------------------
# extract_mstft (Multi-Resolution STFT Distance)
# ---------------------------------------------------------------------------


class TestExtractMstft:
    """Tests for evaluation/metrics/spectrogram/multi_resolution_stft_distance.py."""

    def test_identical_audio_returns_near_zero_distance(self, tmp_path):
        """Identical audio files should produce a multi-resolution STFT distance of ~0.

        When ref == deg, the spectral convergence loss numerator (||mag_ref - mag_deg||_F)
        is 0, and the log-magnitude L1 loss (log(mag_ref) - log(mag_deg)) is also 0.
        The returned sum should be very close to 0.
        """
        from evaluation.metrics.spectrogram.multi_resolution_stft_distance import (
            extract_mstft,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_mstft(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"extract_mstft must return a numeric value, got {type(result)}"
        )
        assert abs(result) < 1e-4, (
            f"MSTFT distance for identical audio should be ~0, got {result}"
        )

    def test_returns_numeric_value(self, tmp_path):
        """extract_mstft must return a single numeric (int or float) value."""
        from evaluation.metrics.spectrogram.multi_resolution_stft_distance import (
            extract_mstft,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_mstft(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"extract_mstft must return a numeric value, got {type(result)}"
        )

    def test_noisy_audio_has_larger_distance_than_identical(self, tmp_path):
        """Noisy degraded audio should produce a larger MSTFT distance than identical audio."""
        from evaluation.metrics.spectrogram.multi_resolution_stft_distance import (
            extract_mstft,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_identical = _make_sine_wav(tmp_path, "deg_identical.wav")
        deg_noisy = _make_noise_wav(tmp_path, "deg_noisy.wav")

        dist_identical = extract_mstft(
            ref_path,
            deg_identical,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )
        dist_noisy = extract_mstft(
            ref_path,
            deg_noisy,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert dist_noisy > dist_identical, (
            f"MSTFT distance for noisy audio ({dist_noisy:.4f}) should exceed "
            f"distance for identical audio ({dist_identical:.4f})"
        )

    def test_no_fs_uses_native_sample_rate(self, tmp_path):
        """Passing fs=None should load audio at its native sample rate without error."""
        from evaluation.metrics.spectrogram.multi_resolution_stft_distance import (
            extract_mstft,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_mstft(
            ref_path,
            deg_path,
            kwargs={"fs": None, "method": "cut"},
        )

        assert isinstance(result, (int, float)), (
            f"extract_mstft with fs=None must return a numeric value, got {type(result)}"
        )

    def test_distance_is_non_negative(self, tmp_path):
        """The MSTFT distance must be non-negative for any pair of audio files."""
        from evaluation.metrics.spectrogram.multi_resolution_stft_distance import (
            extract_mstft,
        )

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_noise_wav(tmp_path, "deg.wav")

        result = extract_mstft(
            ref_path,
            deg_path,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert result >= 0.0, (
            f"MSTFT distance must be non-negative, got {result}"
        )
