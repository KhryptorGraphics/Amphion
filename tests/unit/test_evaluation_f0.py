# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for evaluation/metrics/f0/ modules.

Covers:
- extract_f0rmse: two identical sine waves should return near-0 RMSE
- extract_fpc: identical audio should return correlation near 1.0
- extract_f1_v_uv: voiced/unvoiced F1 score components (TP, FP, FN)

All tests use temporary WAV files created with soundfile.
Tests are skipped if parselmouth is not available.
"""

import sys
import types

import numpy as np
import pytest
import soundfile as sf

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that utils/f0.py imports at module
# level. pyworld and torchcrepe are not needed for f0 evaluation metric tests;
# they are only used by unrelated extractors (dio, harvest, crepe).
# ---------------------------------------------------------------------------
for _mod_name in ("pyworld", "torchcrepe", "torchcrepe.filter", "torchcrepe.threshold"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# Skip the entire module if parselmouth is not installed.
# f0_corr.py imports parselmouth at module level, and all three metric
# functions use get_f0_features_using_parselmouth internally.
pytest.importorskip(
    "parselmouth",
    reason="parselmouth not installed; skipping F0 evaluation metric tests",
)

# ---------------------------------------------------------------------------
# Audio creation helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_DURATION = 2.0  # seconds — long enough for reliable f0 extraction
_FREQ = 200.0    # Hz — well within the default f0 range [50, 1100]


def _make_sine_wav(tmp_path, filename, freq=_FREQ, duration=_DURATION, sample_rate=_SAMPLE_RATE):
    """Create a temporary WAV file with a pure sine wave at a given frequency."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / filename
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


def _make_chirp_wav(
    tmp_path,
    filename,
    f_start=180.0,
    f_end=220.0,
    duration=_DURATION,
    sample_rate=_SAMPLE_RATE,
):
    """Create a temporary WAV file with a linear frequency sweep (chirp).

    Using a chirp ensures the detected f0 array varies over time, making
    Pearson correlation well-defined (avoids NaN from zero-variance arrays).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freq_t = np.linspace(f_start, f_end, len(t))
    phase = np.cumsum(2 * np.pi * freq_t / sample_rate)
    audio = (np.sin(phase) * 0.5).astype(np.float32)
    wav_path = tmp_path / filename
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


# ---------------------------------------------------------------------------
# extract_f0rmse
# ---------------------------------------------------------------------------


class TestExtractF0Rmse:
    """Tests for evaluation/metrics/f0/f0_rmse.py::extract_f0rmse."""

    def test_identical_audio_returns_near_zero_rmse(self, tmp_path):
        """Identical sine waves should produce an F0 RMSE of 0 (or very close).

        Since both files are byte-identical, parselmouth yields identical f0
        arrays and the RMSE is exactly 0.
        """
        from evaluation.metrics.f0.f0_rmse import extract_f0rmse

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        rmse = extract_f0rmse(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut", "need_mean": False},
        )

        assert rmse < 5.0, f"RMSE for identical audio should be near 0, got {rmse}"

    def test_returns_non_negative_float(self, tmp_path):
        """extract_f0rmse must return a non-negative numeric value."""
        from evaluation.metrics.f0.f0_rmse import extract_f0rmse

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        rmse = extract_f0rmse(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut", "need_mean": False},
        )

        assert isinstance(rmse, (int, float)), f"RMSE must be numeric, got {type(rmse)}"
        assert rmse >= 0.0, f"RMSE must be non-negative, got {rmse}"

    def test_different_pitches_give_higher_rmse(self, tmp_path):
        """Audio files with different pitches should yield a higher RMSE than identical ones."""
        from evaluation.metrics.f0.f0_rmse import extract_f0rmse

        # Different-pitch pair
        ref_path = _make_sine_wav(tmp_path, "ref_200.wav", freq=200.0)
        deg_path = _make_sine_wav(tmp_path, "deg_400.wav", freq=400.0)
        rmse_different = extract_f0rmse(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut", "need_mean": False},
        )

        # Identical-pitch pair
        ref_same = _make_sine_wav(tmp_path, "ref_same.wav", freq=200.0)
        deg_same = _make_sine_wav(tmp_path, "deg_same.wav", freq=200.0)
        rmse_same = extract_f0rmse(
            ref_same,
            deg_same,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut", "need_mean": False},
        )

        assert rmse_different > rmse_same, (
            f"RMSE for different pitches ({rmse_different:.2f}) should be greater than "
            f"RMSE for identical audio ({rmse_same:.2f})"
        )

    def test_dtw_alignment_method_identical_audio(self, tmp_path):
        """extract_f0rmse with method='dtw' should also return near-zero for identical files."""
        from evaluation.metrics.f0.f0_rmse import extract_f0rmse

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        rmse = extract_f0rmse(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "dtw", "need_mean": False},
        )

        assert rmse >= 0.0, f"RMSE must be non-negative, got {rmse}"
        assert rmse < 5.0, f"RMSE for identical audio (dtw) should be near 0, got {rmse}"


# ---------------------------------------------------------------------------
# extract_fpc (F0 Pearson Correlation Coefficient)
# ---------------------------------------------------------------------------


class TestExtractFpc:
    """Tests for evaluation/metrics/f0/f0_corr.py::extract_fpc."""

    @pytest.fixture(autouse=True)
    def require_torchmetrics(self):
        """Skip all tests in this class if torchmetrics is not installed.

        extract_fpc imports PearsonCorrCoef from torchmetrics at call time.
        """
        pytest.importorskip("torchmetrics", reason="torchmetrics not installed")

    def test_identical_chirp_audio_returns_correlation_near_one(self, tmp_path):
        """Identical chirp audio should produce a Pearson correlation near 1.0.

        A chirp (frequency sweep from 180 to 220 Hz) guarantees that the
        detected f0 array is non-constant over time, making the Pearson
        correlation well-defined and equal to 1.0 for identical files.
        """
        from evaluation.metrics.f0.f0_corr import extract_fpc

        ref_path = _make_chirp_wav(tmp_path, "ref.wav")
        deg_path = _make_chirp_wav(tmp_path, "deg.wav")

        corr = extract_fpc(
            ref_path,
            deg_path,
            fs=_SAMPLE_RATE,
            need_mean=False,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            method="cut",
        )

        # NaN only occurs when the f0 array has zero variance (constant pitch).
        # For a chirp this should not happen, but skip gracefully if it does.
        if isinstance(corr, float) and np.isnan(corr):
            pytest.skip(
                "f0 arrays have zero variance (constant pitch detected); "
                "Pearson correlation is undefined — not a metric error"
            )

        assert corr > 0.9, (
            f"Pearson correlation for identical chirp audio should be near 1.0, got {corr}"
        )

    def test_returns_numeric_value(self, tmp_path):
        """extract_fpc must return a numeric value (int or float)."""
        from evaluation.metrics.f0.f0_corr import extract_fpc

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        corr = extract_fpc(
            ref_path,
            deg_path,
            fs=_SAMPLE_RATE,
            need_mean=False,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            method="cut",
        )

        assert isinstance(corr, (int, float)), (
            f"Correlation must be a numeric value, got {type(corr)}"
        )

    def test_correlation_in_valid_range_or_nan(self, tmp_path):
        """Non-NaN correlation values must lie in [-1, 1]."""
        from evaluation.metrics.f0.f0_corr import extract_fpc

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        corr = extract_fpc(
            ref_path,
            deg_path,
            fs=_SAMPLE_RATE,
            need_mean=False,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            method="cut",
        )

        # NaN is acceptable for constant f0 arrays (zero variance)
        if not (isinstance(corr, float) and np.isnan(corr)):
            assert -1.0 <= corr <= 1.0, (
                f"Non-NaN correlation must be in [-1, 1], got {corr}"
            )

    def test_extract_fpc_does_not_raise_for_identical_audio(self, tmp_path):
        """extract_fpc must complete without raising an exception for identical files."""
        from evaluation.metrics.f0.f0_corr import extract_fpc

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        # Should not raise any exception
        corr = extract_fpc(
            ref_path,
            deg_path,
            fs=_SAMPLE_RATE,
            need_mean=False,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            method="cut",
        )
        assert corr is not None


# ---------------------------------------------------------------------------
# extract_f1_v_uv (Voiced/Unvoiced F1 Score)
# ---------------------------------------------------------------------------


class TestExtractF1VUv:
    """Tests for evaluation/metrics/f0/v_uv_f1.py::extract_f1_v_uv."""

    def test_returns_three_numeric_values(self, tmp_path):
        """extract_f1_v_uv must return a 3-tuple of numeric (tp, fp, fn) values."""
        from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        result = extract_f1_v_uv(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert len(result) == 3, (
            f"extract_f1_v_uv must return a 3-tuple, got length {len(result)}"
        )
        tp, fp, fn = result
        assert isinstance(tp, (int, float)), f"TP must be numeric, got {type(tp)}"
        assert isinstance(fp, (int, float)), f"FP must be numeric, got {type(fp)}"
        assert isinstance(fn, (int, float)), f"FN must be numeric, got {type(fn)}"

    def test_identical_voiced_audio_zero_fp_and_fn(self, tmp_path):
        """Identical voiced audio: FP and FN must both be 0.

        For identical files the voiced/unvoiced mask is the same for ref and
        deg, so there are no false positives or false negatives.
        """
        from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        tp, fp, fn = extract_f1_v_uv(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert fp == 0, f"FP should be 0 for identical audio, got {fp}"
        assert fn == 0, f"FN should be 0 for identical audio, got {fn}"

    def test_voiced_sine_wave_has_positive_tp(self, tmp_path):
        """For voiced sine wave audio, at least some frames must be voiced (TP > 0)."""
        from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        tp, fp, fn = extract_f1_v_uv(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert tp > 0, f"For voiced audio, TP should be > 0, got {tp}"

    def test_all_counts_non_negative(self, tmp_path):
        """TP, FP, and FN must all be non-negative."""
        from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        tp, fp, fn = extract_f1_v_uv(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "cut"},
        )

        assert tp >= 0, f"TP must be non-negative, got {tp}"
        assert fp >= 0, f"FP must be non-negative, got {fp}"
        assert fn >= 0, f"FN must be non-negative, got {fn}"

    def test_dtw_alignment_does_not_raise(self, tmp_path):
        """extract_f1_v_uv with method='dtw' should complete without error."""
        from evaluation.metrics.f0.v_uv_f1 import extract_f1_v_uv

        ref_path = _make_sine_wav(tmp_path, "ref.wav")
        deg_path = _make_sine_wav(tmp_path, "deg.wav")

        tp, fp, fn = extract_f1_v_uv(
            ref_path,
            deg_path,
            hop_length=256,
            f0_min=50,
            f0_max=1100,
            kwargs={"fs": _SAMPLE_RATE, "method": "dtw"},
        )

        assert tp >= 0, f"TP must be non-negative, got {tp}"
        assert fp >= 0, f"FP must be non-negative, got {fp}"
        assert fn >= 0, f"FN must be non-negative, got {fn}"
