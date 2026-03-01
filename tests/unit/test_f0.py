# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for utils/f0.py

Covers pure helper functions only:
- f0_to_coarse: returns int array with values in [1, pitch_bin-1]
- interpolate: unvoiced frames (f0==0) receive interpolated values
- get_log_f0: zero frames become 0 (log(1)), others get log applied
- get_pitch_sub_median: output has zero median (in cents space)

Skipped (integration tests requiring audio files / system libraries):
- get_f0_features_using_pyin      (requires librosa.pyin + audio)
- get_f0_features_using_parselmouth (requires parselmouth + audio)
- get_f0_features_using_dio        (requires pyworld + audio)
- get_f0_features_using_harvest    (requires pyworld + audio)
- get_f0_features_using_crepe      (requires torchcrepe + GPU + audio)
"""

import sys
import types
import unittest.mock as mock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies that utils/f0.py imports at module
# level.  pyworld, parselmouth, and torchcrepe are NOT needed for the pure
# helper functions we test here; they are only used by the integration-only
# extractors (get_f0_features_using_dio, _harvest, _parselmouth, _crepe).
# ---------------------------------------------------------------------------
for _mod_name in ("pyworld", "parselmouth", "torchcrepe",
                  "torchcrepe.filter", "torchcrepe.threshold"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

from utils.f0 import (  # noqa: E402  (import after sys.modules manipulation)
    f0_to_coarse,
    get_log_f0,
    get_pitch_sub_median,
    interpolate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Standard parameters used across tests
_PITCH_BIN = 256
_F0_MIN = 50.0
_F0_MAX = 1100.0


def _make_voiced_f0(n=20, f0_min=100.0, f0_max=400.0, seed=0):
    """Return a float32 numpy array of fully voiced f0 values (all > 0)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(f0_min, f0_max, size=n).astype(np.float32)


def _make_mixed_f0(n=20, unvoiced_indices=(0, 5, 10), voiced_f0_val=220.0):
    """Return a float32 numpy array with some unvoiced frames (f0 == 0)."""
    f0 = np.full(n, voiced_f0_val, dtype=np.float32)
    for idx in unvoiced_indices:
        f0[idx] = 0.0
    return f0


# ---------------------------------------------------------------------------
# f0_to_coarse
# ---------------------------------------------------------------------------


class TestF0ToCoarse:
    def test_returns_int32_numpy_array(self):
        """f0_to_coarse on a numpy input must return a numpy int32 array."""
        f0 = _make_voiced_f0()
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert isinstance(result, np.ndarray), "Result must be a numpy array"
        assert result.dtype == np.int32, f"Expected int32, got {result.dtype}"

    def test_values_in_valid_range_numpy(self):
        """All coarse values must lie within [1, pitch_bin - 1]."""
        f0 = _make_voiced_f0()
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert result.min() >= 1, f"Min coarse value {result.min()} < 1"
        assert result.max() <= _PITCH_BIN - 1, (
            f"Max coarse value {result.max()} > {_PITCH_BIN - 1}"
        )

    def test_zero_f0_maps_to_bin_1(self):
        """Unvoiced frames (f0==0) produce mel=0 <= 1, so they map to bin 1."""
        f0 = np.array([0.0, 220.0, 0.0], dtype=np.float32)
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert result[0] == 1, f"Zero f0 should map to bin 1, got {result[0]}"
        assert result[2] == 1, f"Zero f0 should map to bin 1, got {result[2]}"

    def test_f0_at_max_clips_to_pitch_bin_minus_1(self):
        """f0 values at or above f0_max should clip to pitch_bin - 1."""
        f0 = np.array([_F0_MAX, _F0_MAX * 2], dtype=np.float32)
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert result.max() == _PITCH_BIN - 1, (
            f"Expected max bin {_PITCH_BIN - 1}, got {result.max()}"
        )

    def test_output_shape_matches_input(self):
        """Output array shape must match input array shape."""
        f0 = _make_voiced_f0(n=50)
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert result.shape == f0.shape, (
            f"Shape mismatch: got {result.shape}, expected {f0.shape}"
        )

    def test_returns_long_tensor_for_torch_input(self):
        """When the input is a torch.Tensor, the output must be a long tensor."""
        f0 = torch.tensor(_make_voiced_f0(), dtype=torch.float32)
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert isinstance(result, torch.Tensor), "Result must be a torch.Tensor"
        assert result.dtype == torch.long, f"Expected long, got {result.dtype}"

    def test_torch_values_in_valid_range(self):
        """Torch-based coarse values must lie within [1, pitch_bin - 1]."""
        f0 = torch.tensor(_make_voiced_f0(), dtype=torch.float32)
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert result.min().item() >= 1
        assert result.max().item() <= _PITCH_BIN - 1

    def test_higher_f0_yields_higher_bin(self):
        """A higher f0 within range should produce a higher or equal coarse bin."""
        f0_low = np.array([100.0], dtype=np.float32)
        f0_high = np.array([800.0], dtype=np.float32)
        result_low = f0_to_coarse(f0_low, _PITCH_BIN, _F0_MIN, _F0_MAX)
        result_high = f0_to_coarse(f0_high, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert result_high[0] >= result_low[0], (
            f"Expected bin({f0_high[0]}) >= bin({f0_low[0]}), "
            f"got {result_high[0]} vs {result_low[0]}"
        )

    def test_no_nan_or_inf_in_output(self):
        """Coarse bins must all be finite integers (no NaN / Inf)."""
        f0 = _make_voiced_f0(n=100)
        result = f0_to_coarse(f0, _PITCH_BIN, _F0_MIN, _F0_MAX)

        assert np.all(np.isfinite(result.astype(np.float32))), (
            "All coarse bins must be finite"
        )


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------


class TestInterpolate:
    def test_returns_tuple_of_two(self):
        """interpolate must return a 2-tuple (f0, uv)."""
        f0 = _make_mixed_f0()
        result = interpolate(f0)

        assert isinstance(result, tuple) and len(result) == 2, (
            "interpolate must return a 2-element tuple"
        )

    def test_unvoiced_frames_filled_after_interpolation(self):
        """After interpolation, previously unvoiced frames should be non-zero."""
        unvoiced = (2, 5)
        f0 = _make_mixed_f0(n=15, unvoiced_indices=unvoiced, voiced_f0_val=200.0)
        f0_interp, _ = interpolate(f0.copy())

        for idx in unvoiced:
            assert f0_interp[idx] != 0.0, (
                f"Frame {idx} should be interpolated (non-zero), got {f0_interp[idx]}"
            )

    def test_voiced_frames_unchanged(self):
        """Voiced frames should keep their original values after interpolation."""
        f0 = _make_mixed_f0(n=15, unvoiced_indices=(0, 14), voiced_f0_val=300.0)
        f0_orig = f0.copy()
        f0_interp, _ = interpolate(f0)

        # Middle voiced frames (index 1..13) should remain 300.0
        for idx in range(1, 14):
            assert f0_interp[idx] == pytest.approx(f0_orig[idx]), (
                f"Voiced frame {idx} changed from {f0_orig[idx]} to {f0_interp[idx]}"
            )

    def test_uv_shape_matches_f0(self):
        """The returned uv array must have the same shape as the input f0."""
        f0 = _make_mixed_f0(n=20)
        f0_interp, uv = interpolate(f0.copy())

        assert uv.shape == f0_interp.shape, (
            f"uv shape {uv.shape} must match f0 shape {f0_interp.shape}"
        )

    def test_fully_voiced_returns_unchanged_f0(self):
        """If all frames are voiced, f0 values must be preserved."""
        f0 = _make_voiced_f0(n=10, f0_min=100.0, f0_max=400.0)
        f0_orig = f0.copy()
        f0_interp, _ = interpolate(f0)

        np.testing.assert_array_almost_equal(
            f0_interp, f0_orig, err_msg="Fully voiced f0 must be unchanged"
        )

    def test_fully_unvoiced_does_not_crash(self):
        """An entirely silent (all-zero) f0 sequence must not raise an error."""
        f0 = np.zeros(10, dtype=np.float32)
        f0_interp, uv = interpolate(f0)  # should not raise

        assert f0_interp.shape == (10,)

    def test_interpolated_values_within_voiced_range(self):
        """Interpolated f0 values should lie between the min and max voiced frames."""
        f0 = np.array([0.0, 100.0, 0.0, 200.0, 0.0], dtype=np.float32)
        f0_interp, _ = interpolate(f0.copy())

        voiced_min = 100.0
        voiced_max = 200.0
        for idx in (0, 2, 4):
            assert f0_interp[idx] >= voiced_min - 1e-3
            assert f0_interp[idx] <= voiced_max + 1e-3


# ---------------------------------------------------------------------------
# get_log_f0
# ---------------------------------------------------------------------------


class TestGetLogF0:
    def test_output_shape_matches_input(self):
        """Output of get_log_f0 must have the same shape as the input."""
        f0 = _make_voiced_f0(n=30)
        log_f0 = get_log_f0(f0.copy())

        assert log_f0.shape == f0.shape

    def test_zero_frames_become_zero_in_log_space(self):
        """Unvoiced (0 Hz) frames are replaced by 1 Hz before log, so log(1) == 0."""
        f0 = np.array([0.0, 220.0, 0.0], dtype=np.float32)
        log_f0 = get_log_f0(f0.copy())

        assert log_f0[0] == pytest.approx(0.0, abs=1e-6), (
            f"Zero frame should give log(1)=0, got {log_f0[0]}"
        )
        assert log_f0[2] == pytest.approx(0.0, abs=1e-6), (
            f"Zero frame should give log(1)=0, got {log_f0[2]}"
        )

    def test_nonzero_frames_are_log_of_input(self):
        """Voiced frames must equal np.log of the original f0 value."""
        f0 = np.array([100.0, 220.0, 440.0], dtype=np.float32)
        f0_orig = f0.copy()
        log_f0 = get_log_f0(f0.copy())

        expected = np.log(f0_orig)
        np.testing.assert_allclose(log_f0, expected, rtol=1e-5)

    def test_all_positive_voiced_yields_positive_log_above_1hz(self):
        """For f0 values > 1 Hz, log_f0 must be positive."""
        f0 = np.array([100.0, 220.0, 440.0, 880.0], dtype=np.float32)
        log_f0 = get_log_f0(f0.copy())

        assert np.all(log_f0 > 0.0), "log(f0) for f0 > 1 Hz must be positive"

    def test_output_finite(self):
        """All output values must be finite."""
        f0 = _make_voiced_f0(n=50)
        log_f0 = get_log_f0(f0.copy())

        assert np.all(np.isfinite(log_f0)), "All log_f0 values must be finite"

    def test_output_monotone_with_f0(self):
        """Higher f0 values should produce higher log_f0 values (log is monotone)."""
        f0 = np.array([100.0, 200.0, 400.0], dtype=np.float32)
        log_f0 = get_log_f0(f0.copy())

        assert log_f0[0] < log_f0[1] < log_f0[2], (
            "log_f0 must be monotonically increasing with f0"
        )


# ---------------------------------------------------------------------------
# get_pitch_sub_median
# ---------------------------------------------------------------------------


class TestGetPitchSubMedian:
    def test_output_has_zero_median(self):
        """The median of the output should be (approximately) zero."""
        f0 = _make_voiced_f0(n=20, f0_min=100.0, f0_max=500.0)
        result = get_pitch_sub_median(f0.copy())

        assert abs(np.median(result)) < 1e-4, (
            f"Median of output should be 0, got {np.median(result)}"
        )

    def test_output_length_equals_voiced_frame_count(self):
        """Output length equals the number of voiced (non-zero) frames."""
        f0 = np.array([0.0, 100.0, 200.0, 0.0, 300.0], dtype=np.float32)
        voiced_count = np.sum(f0 != 0)
        result = get_pitch_sub_median(f0.copy())

        assert result.shape[0] == voiced_count, (
            f"Expected {voiced_count} voiced frames, got {result.shape[0]}"
        )

    def test_output_is_in_cents_space(self):
        """Output should be in cents: plausible range for typical singing/speech."""
        # Cents range for speech: roughly +-600 cents (one octave up/down)
        f0 = _make_voiced_f0(n=30, f0_min=100.0, f0_max=400.0)
        result = get_pitch_sub_median(f0.copy())

        # Cents deviation from median must be within a wide but finite bound
        assert np.all(np.abs(result) < 3000.0), (
            "Median-subtracted cents should be within +-3000 cents"
        )

    def test_output_finite(self):
        """All output values must be finite (no NaN or Inf)."""
        f0 = _make_voiced_f0(n=20, f0_min=100.0, f0_max=500.0)
        result = get_pitch_sub_median(f0.copy())

        assert np.all(np.isfinite(result)), "get_pitch_sub_median output must be finite"

    def test_constant_f0_gives_all_zeros(self):
        """If all voiced frames have the same pitch, median-subtracted result is zero."""
        f0 = np.full(10, 220.0, dtype=np.float32)
        result = get_pitch_sub_median(f0)

        np.testing.assert_allclose(
            result,
            np.zeros_like(result),
            atol=1e-4,
            err_msg="Constant pitch should give zero after median subtraction",
        )

    def test_positive_and_negative_deviations_present(self):
        """For a varied f0, result must contain both positive and negative values."""
        f0 = np.array([100.0, 150.0, 200.0, 250.0, 300.0], dtype=np.float32)
        result = get_pitch_sub_median(f0)

        assert np.any(result > 0), "Some cents deviations should be positive"
        assert np.any(result < 0), "Some cents deviations should be negative"


# ---------------------------------------------------------------------------
# Integration test markers (skipped in unit test runs)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Integration test: requires real audio file and librosa.pyin. "
        "Run with pytest -m integration."
    )
)
def test_get_f0_features_using_pyin_integration():
    """Placeholder to document pyin as an integration test."""
    pass


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Integration test: requires real audio file and parselmouth. "
        "Run with pytest -m integration."
    )
)
def test_get_f0_features_using_parselmouth_integration():
    """Placeholder to document parselmouth as an integration test."""
    pass


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Integration test: requires real audio file and pyworld. "
        "Run with pytest -m integration."
    )
)
def test_get_f0_features_using_dio_integration():
    """Placeholder to document dio as an integration test."""
    pass


@pytest.mark.integration
@pytest.mark.skip(
    reason=(
        "Integration test: requires real audio file and torchcrepe + GPU. "
        "Run with pytest -m integration."
    )
)
def test_get_f0_features_using_crepe_integration():
    """Placeholder to document crepe as an integration test."""
    pass
