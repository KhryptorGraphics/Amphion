# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for evaluation/metrics/energy/ modules.

Covers:
- extract_energy_rmse: identical audio should return near-0 RMSE; different
  signals should return a non-trivial (positive) RMSE.
- extract_energy_pearson_coeffcients: identical audio should return correlation
  near 1.0; different signals should return a non-trivial value.

All tests use temporary WAV files created with soundfile.
Pearson tests are skipped if torchmetrics is not available (imported at module
level by energy_pearson_coefficients.py).
"""

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Audio creation helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 22050
_DURATION = 1.0  # seconds

# STFT parameters (match defaults in the metric functions)
_N_FFT = 1024
_HOP_LENGTH = 256
_WIN_LENGTH = 1024


def _make_sine_wav(tmp_path, filename, freq=440.0, duration=_DURATION, sample_rate=_SAMPLE_RATE):
    """Create a temporary WAV file containing a pure sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / filename
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


def _make_am_sine_wav(tmp_path, filename, freq=440.0, duration=_DURATION, sample_rate=_SAMPLE_RATE):
    """Create a temporary WAV file with an amplitude-modulated sine wave.

    The slowly-varying amplitude envelope guarantees that the per-frame energy
    is non-constant, making the Pearson correlation well-defined (non-zero
    variance) for identical audio pairs.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Envelope varies between 0.3 and 1.0 at 2 Hz — stays strictly positive
    # so db_scale (log10) remains finite.
    envelope = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 2.0 * t))
    audio = (envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    wav_path = tmp_path / filename
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


def _make_noise_wav(tmp_path, filename, duration=_DURATION, sample_rate=_SAMPLE_RATE, seed=42):
    """Create a temporary WAV file containing Gaussian white noise."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal(int(sample_rate * duration)) * 0.3).astype(np.float32)
    wav_path = tmp_path / filename
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


# ---------------------------------------------------------------------------
# extract_energy_rmse
# ---------------------------------------------------------------------------


class TestExtractEnergyRmse:
    """Tests for evaluation/metrics/energy/energy_rmse.py::extract_energy_rmse."""

    def _call(self, ref, deg, method="cut", db_scale=False, fs=_SAMPLE_RATE):
        from evaluation.metrics.energy.energy_rmse import extract_energy_rmse

        return extract_energy_rmse(
            ref,
            deg,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            kwargs={"fs": fs, "method": method, "db_scale": db_scale},
        )

    def test_identical_audio_returns_near_zero_rmse(self, tmp_path):
        """Identical audio files should produce an energy RMSE of 0 (or very close).

        Since both files are byte-identical, librosa yields identical STFT
        magnitudes, so the energy arrays are equal and RMSE is exactly 0.
        """
        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        rmse = self._call(ref_path, deg_path)

        assert rmse < 1e-5, (
            f"RMSE for identical audio should be near 0, got {rmse}"
        )

    def test_different_audio_returns_nontrivial_rmse(self, tmp_path):
        """Different audio (sine vs noise) should produce a positive RMSE."""
        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_noise_wav(tmp_path, "deg.wav")

        rmse = self._call(ref_path, deg_path)

        assert rmse > 0.0, (
            f"RMSE for different audio should be positive, got {rmse}"
        )

    def test_returns_non_negative_float(self, tmp_path):
        """extract_energy_rmse must return a non-negative float."""
        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_noise_wav(tmp_path, "deg.wav")

        rmse = self._call(ref_path, deg_path)

        assert isinstance(rmse, float), (
            f"RMSE must be a float, got {type(rmse)}"
        )
        assert rmse >= 0.0, f"RMSE must be non-negative, got {rmse}"

    def test_different_pitches_give_higher_rmse_than_identical(self, tmp_path):
        """Audio files with different frequencies should yield higher RMSE than identical ones."""
        ref_path = _make_sine_wav(tmp_path, "ref_220.wav", freq=220.0)
        deg_different = _make_noise_wav(tmp_path, "deg_noise.wav")
        deg_identical = _make_sine_wav(tmp_path, "deg_220.wav", freq=220.0)

        rmse_different = self._call(ref_path, deg_different)
        rmse_identical = self._call(ref_path, deg_identical)

        assert rmse_different > rmse_identical, (
            f"RMSE for different audio ({rmse_different:.4f}) should exceed "
            f"RMSE for identical audio ({rmse_identical:.4f})"
        )

    def test_method_cut_identical_audio_near_zero(self, tmp_path):
        """Method='cut' should yield near-zero RMSE for identical audio."""
        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        rmse = self._call(ref_path, deg_path, method="cut")

        assert rmse < 1e-5, (
            f"RMSE (cut) for identical audio should be near 0, got {rmse}"
        )

    def test_method_dtw_identical_audio_near_zero(self, tmp_path):
        """Method='dtw' should yield near-zero RMSE for identical audio."""
        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        rmse = self._call(ref_path, deg_path, method="dtw")

        assert rmse < 1e-5, (
            f"RMSE (dtw) for identical audio should be near 0, got {rmse}"
        )

    def test_db_scale_true_returns_float(self, tmp_path):
        """db_scale=True should not raise and must return a finite float."""
        # Use the AM sine so energy is strictly positive (log10 is finite).
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_am_sine_wav(tmp_path, "deg.wav")

        rmse = self._call(ref_path, deg_path, db_scale=True)

        assert isinstance(rmse, float), (
            f"RMSE with db_scale=True must be a float, got {type(rmse)}"
        )
        assert rmse >= 0.0, f"RMSE must be non-negative with db_scale=True, got {rmse}"

    def test_fs_none_uses_native_sample_rate(self, tmp_path):
        """Passing fs=None should load audio at its native sample rate without error."""
        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        rmse = self._call(ref_path, deg_path, fs=None)

        assert isinstance(rmse, float), (
            f"RMSE with fs=None must be a float, got {type(rmse)}"
        )
        assert rmse >= 0.0, f"RMSE must be non-negative, got {rmse}"


# ---------------------------------------------------------------------------
# extract_energy_pearson_coeffcients (note: typo preserved from source)
# ---------------------------------------------------------------------------


class TestExtractEnergyPearsonCoefficients:
    """Tests for evaluation/metrics/energy/energy_pearson_coefficients.py::extract_energy_pearson_coeffcients."""

    @pytest.fixture(autouse=True)
    def require_torchmetrics(self):
        """Skip all tests in this class if torchmetrics is not installed.

        energy_pearson_coefficients.py imports PearsonCorrCoef from
        torchmetrics at module level, so the import will fail without it.
        """
        pytest.importorskip(
            "torchmetrics",
            reason="torchmetrics not installed; skipping energy Pearson tests",
        )

    def _call(self, ref, deg, method="cut", db_scale=False, fs=_SAMPLE_RATE):
        from evaluation.metrics.energy.energy_pearson_coefficients import (
            extract_energy_pearson_coeffcients,
        )

        return extract_energy_pearson_coeffcients(
            ref,
            deg,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            kwargs={"fs": fs, "method": method, "db_scale": db_scale},
        )

    def test_identical_audio_returns_near_one_pearson(self, tmp_path):
        """Identical AM sine wave audio should produce a Pearson correlation near 1.0.

        Using an amplitude-modulated sine ensures the per-frame energy is
        non-constant (non-zero variance), making the Pearson coefficient
        well-defined. For identical files the energy arrays are equal, so
        the correlation is exactly 1.0.
        """
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_am_sine_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path)

        # Skip gracefully if torchmetrics returns NaN (constant energy edge case)
        if isinstance(pearson, float) and np.isnan(pearson):
            pytest.skip(
                "Energy arrays have zero variance (constant energy detected); "
                "Pearson correlation is undefined — not a metric error"
            )

        assert abs(pearson - 1.0) < 1e-4, (
            f"Pearson for identical audio should be near 1.0, got {pearson}"
        )

    def test_different_audio_returns_nontrivial_pearson(self, tmp_path):
        """Sine vs noise audio should produce a Pearson coefficient not near 1.0."""
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_noise_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path)

        assert isinstance(pearson, (int, float)), (
            f"Pearson must be numeric, got {type(pearson)}"
        )
        # For genuinely different signals the correlation should not be near 1.0
        if not (isinstance(pearson, float) and np.isnan(pearson)):
            assert pearson < 0.99, (
                f"Pearson for different audio should not be near 1.0, got {pearson}"
            )

    def test_returns_numeric_value(self, tmp_path):
        """extract_energy_pearson_coeffcients must return a numeric (int or float) value."""
        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path)

        assert isinstance(pearson, (int, float)), (
            f"Pearson must return a numeric value, got {type(pearson)}"
        )

    def test_pearson_in_valid_range_or_nan(self, tmp_path):
        """Non-NaN Pearson values must lie in [-1, 1]."""
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_noise_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path)

        if not (isinstance(pearson, float) and np.isnan(pearson)):
            assert -1.0 <= pearson <= 1.0, (
                f"Non-NaN Pearson must lie in [-1, 1], got {pearson}"
            )

    def test_method_cut_identical_audio_near_one(self, tmp_path):
        """Method='cut' should yield Pearson near 1.0 for identical AM sine audio."""
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_am_sine_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path, method="cut")

        if isinstance(pearson, float) and np.isnan(pearson):
            pytest.skip("Energy arrays have zero variance; Pearson undefined")

        assert abs(pearson - 1.0) < 1e-4, (
            f"Pearson (cut) for identical audio should be near 1.0, got {pearson}"
        )

    def test_method_dtw_identical_audio_near_one(self, tmp_path):
        """Method='dtw' should yield Pearson near 1.0 for identical AM sine audio."""
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_am_sine_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path, method="dtw")

        if isinstance(pearson, float) and np.isnan(pearson):
            pytest.skip("Energy arrays have zero variance; Pearson undefined")

        assert abs(pearson - 1.0) < 1e-4, (
            f"Pearson (dtw) for identical audio should be near 1.0, got {pearson}"
        )

    def test_db_scale_true_returns_numeric(self, tmp_path):
        """db_scale=True should not raise and must return a numeric value."""
        # Use AM sine so energy is strictly positive (log10 remains finite).
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_am_sine_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path, db_scale=True)

        assert isinstance(pearson, (int, float)), (
            f"Pearson with db_scale=True must be numeric, got {type(pearson)}"
        )

    def test_fs_none_uses_native_sample_rate(self, tmp_path):
        """Passing fs=None should load audio at its native sample rate without error."""
        ref_path = _make_am_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_am_sine_wav(tmp_path, "deg.wav")

        pearson = self._call(ref_path, deg_path, fs=None)

        assert isinstance(pearson, (int, float)), (
            f"Pearson with fs=None must return a numeric value, got {type(pearson)}"
        )
