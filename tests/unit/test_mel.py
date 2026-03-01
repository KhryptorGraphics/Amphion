# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for utils/mel.py

Covers:
- dynamic_range_compression_torch: log compression and clip behavior
- spectral_normalize_torch: output shape and value correctness
- extract_linear_features: output shape [n_fft//2+1, T]
- extract_mel_features: output shape [n_mel, T] and values < 0 (log-mel)
- mel_spectrogram_torch: output shape [n_mel, T]
- amplitude_phase_spectrum: log_amplitude and phase output shapes match
"""

import math

import pytest
import torch

import utils.mel as mel_module
from utils.mel import (
    amplitude_phase_spectrum,
    dynamic_range_compression_torch,
    extract_linear_features,
    extract_mel_features,
    mel_spectrogram_torch,
    spectral_normalize_torch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio(num_samples: int = 24000, batch: int = 1) -> torch.Tensor:
    """Return a float32 audio tensor of shape [batch, num_samples]."""
    torch.manual_seed(42)
    return torch.randn(batch, num_samples).clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# dynamic_range_compression_torch
# ---------------------------------------------------------------------------


class TestDynamicRangeCompressionTorch:
    def test_output_shape_preserved(self):
        x = torch.ones(4, 80, 100)
        out = dynamic_range_compression_torch(x)
        assert out.shape == x.shape

    def test_log_compression_positive_values(self):
        """log(C * x) with C=1 should equal torch.log(x) for x > clip_val."""
        x = torch.tensor([1.0, math.e, math.e ** 2])
        out = dynamic_range_compression_torch(x, C=1, clip_val=1e-5)
        expected = torch.log(x)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_clip_behavior_below_clip_val(self):
        """Values below clip_val should be clipped, not log(0) = -inf."""
        clip_val = 1e-5
        x = torch.tensor([0.0, 1e-10, clip_val / 2])
        out = dynamic_range_compression_torch(x, C=1, clip_val=clip_val)
        # All should produce log(clip_val) = -11.5129...
        expected_val = math.log(clip_val)
        assert torch.all(torch.isfinite(out)), "Output should be finite, not -inf"
        assert torch.allclose(out, torch.full_like(out, expected_val), atol=1e-4)

    def test_clip_behavior_at_clip_val(self):
        """Value exactly at clip_val should remain unchanged (clamp does nothing)."""
        clip_val = 1e-5
        x = torch.tensor([clip_val])
        out = dynamic_range_compression_torch(x, C=1, clip_val=clip_val)
        expected = torch.log(torch.tensor([clip_val]))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_scaling_constant_C(self):
        """C multiplier scales input before log: log(C * x)."""
        x = torch.tensor([1.0])
        C = 2.0
        out = dynamic_range_compression_torch(x, C=C, clip_val=1e-10)
        expected = math.log(C * 1.0)
        assert abs(out.item() - expected) < 1e-6

    def test_no_nan_or_inf_in_typical_range(self):
        """With typical positive inputs there should be no NaN or Inf."""
        x = torch.rand(16, 80, 200) + 1e-5  # ensure above clip_val
        out = dynamic_range_compression_torch(x)
        assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# spectral_normalize_torch
# ---------------------------------------------------------------------------


class TestSpectralNormalizeTorch:
    def test_output_shape_matches_input(self):
        magnitudes = torch.rand(80, 200) + 1e-5
        out = spectral_normalize_torch(magnitudes)
        assert out.shape == magnitudes.shape

    def test_is_equivalent_to_dynamic_range_compression(self):
        """spectral_normalize_torch is a thin wrapper around DRC."""
        magnitudes = torch.rand(4, 80, 100) + 1e-5
        out_normalize = spectral_normalize_torch(magnitudes)
        out_drc = dynamic_range_compression_torch(magnitudes)
        assert torch.allclose(out_normalize, out_drc)

    def test_output_finite(self):
        magnitudes = torch.rand(80, 100) + 1e-5
        out = spectral_normalize_torch(magnitudes)
        assert torch.all(torch.isfinite(out))


# ---------------------------------------------------------------------------
# extract_linear_features
# ---------------------------------------------------------------------------


class TestExtractLinearFeatures:
    def test_output_shape_freq_dim(self, basic_mel_cfg):
        """First dimension must equal n_fft // 2 + 1."""
        y = _make_audio(num_samples=24000, batch=1)
        # Clear cached windows to avoid device mismatch across tests
        mel_module.hann_window.clear()

        spec = extract_linear_features(y, basic_mel_cfg)

        expected_freq_bins = basic_mel_cfg.n_fft // 2 + 1  # 513
        assert spec.ndim == 2, f"Expected 2D output, got {spec.ndim}D"
        assert spec.shape[0] == expected_freq_bins, (
            f"Expected freq bins {expected_freq_bins}, got {spec.shape[0]}"
        )

    def test_output_shape_time_dim(self, basic_mel_cfg):
        """Time dimension should be positive and roughly num_samples // hop_size."""
        y = _make_audio(num_samples=24000, batch=1)
        mel_module.hann_window.clear()

        spec = extract_linear_features(y, basic_mel_cfg)

        # Time frames should be in a reasonable range
        assert spec.shape[1] > 0

    def test_output_values_finite_and_nonnegative(self, basic_mel_cfg):
        """Linear spectrogram magnitudes are nonneg (sqrt of sum-of-squares)."""
        y = _make_audio(num_samples=8000, batch=1)
        mel_module.hann_window.clear()

        spec = extract_linear_features(y, basic_mel_cfg)

        assert torch.all(torch.isfinite(spec))
        assert torch.all(spec >= 0.0)


# ---------------------------------------------------------------------------
# extract_mel_features
# ---------------------------------------------------------------------------


class TestExtractMelFeatures:
    def test_output_shape(self, basic_mel_cfg):
        """Output shape must be [n_mel, T]."""
        y = _make_audio(num_samples=24000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = extract_mel_features(y, basic_mel_cfg)

        assert spec.ndim == 2, f"Expected 2D output, got {spec.ndim}D"
        assert spec.shape[0] == basic_mel_cfg.n_mel, (
            f"Expected n_mel={basic_mel_cfg.n_mel}, got {spec.shape[0]}"
        )

    def test_values_negative_for_log_mel(self, basic_mel_cfg):
        """Log-mel spectrogram values should be negative (since magnitudes < 1)."""
        y = _make_audio(num_samples=24000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = extract_mel_features(y, basic_mel_cfg)

        # Most values in a log-mel spectrogram are negative
        assert torch.mean((spec < 0).float()) > 0.5, (
            "Majority of log-mel values should be negative"
        )

    def test_output_finite(self, basic_mel_cfg):
        y = _make_audio(num_samples=8000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = extract_mel_features(y, basic_mel_cfg)

        assert torch.all(torch.isfinite(spec))

    def test_time_dim_positive(self, basic_mel_cfg):
        y = _make_audio(num_samples=24000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = extract_mel_features(y, basic_mel_cfg)

        assert spec.shape[1] > 0


# ---------------------------------------------------------------------------
# mel_spectrogram_torch
# ---------------------------------------------------------------------------


class TestMelSpectrogramTorch:
    def test_output_shape_contains_n_mel(self, basic_mel_cfg):
        """The mel-frequency dimension must equal n_mel."""
        y = _make_audio(num_samples=24000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = mel_spectrogram_torch(y, basic_mel_cfg)

        # With batch=1, output is [1, n_mel, T]
        assert spec.shape[-2] == basic_mel_cfg.n_mel, (
            f"Expected n_mel={basic_mel_cfg.n_mel}, got {spec.shape[-2]}"
        )

    def test_output_is_log_mel(self, basic_mel_cfg):
        """Log-mel values should mostly be negative."""
        y = _make_audio(num_samples=24000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = mel_spectrogram_torch(y, basic_mel_cfg)

        assert torch.mean((spec < 0).float()) > 0.5

    def test_output_finite(self, basic_mel_cfg):
        y = _make_audio(num_samples=8000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = mel_spectrogram_torch(y, basic_mel_cfg)

        assert torch.all(torch.isfinite(spec))

    def test_time_dim_positive(self, basic_mel_cfg):
        y = _make_audio(num_samples=24000, batch=1)
        mel_module.mel_basis.clear()
        mel_module.hann_window.clear()

        spec = mel_spectrogram_torch(y, basic_mel_cfg)

        assert spec.shape[-1] > 0


# ---------------------------------------------------------------------------
# amplitude_phase_spectrum
# ---------------------------------------------------------------------------


class TestAmplitudePhaseSpectrum:
    def test_log_amplitude_and_phase_shapes_match(self, basic_mel_cfg):
        """log_amplitude and phase must have the same shape."""
        y = _make_audio(num_samples=24000, batch=1)

        log_amplitude, phase, rea, imag = amplitude_phase_spectrum(y, basic_mel_cfg)

        assert log_amplitude.shape == phase.shape, (
            f"log_amplitude shape {log_amplitude.shape} != phase shape {phase.shape}"
        )

    def test_output_freq_dim(self, basic_mel_cfg):
        """First dimension should be n_fft // 2 + 1."""
        y = _make_audio(num_samples=24000, batch=1)

        log_amplitude, phase, rea, imag = amplitude_phase_spectrum(y, basic_mel_cfg)

        expected_freq_bins = basic_mel_cfg.n_fft // 2 + 1
        assert log_amplitude.shape[0] == expected_freq_bins, (
            f"Expected freq bins {expected_freq_bins}, got {log_amplitude.shape[0]}"
        )

    def test_phase_range(self, basic_mel_cfg):
        """Phase values (atan2) must lie in [-pi, pi]."""
        y = _make_audio(num_samples=24000, batch=1)

        log_amplitude, phase, rea, imag = amplitude_phase_spectrum(y, basic_mel_cfg)

        assert torch.all(phase >= -math.pi - 1e-6)
        assert torch.all(phase <= math.pi + 1e-6)

    def test_log_amplitude_finite(self, basic_mel_cfg):
        """log_amplitude should be finite (the +1e-5 epsilon prevents log(0))."""
        y = _make_audio(num_samples=24000, batch=1)

        log_amplitude, phase, rea, imag = amplitude_phase_spectrum(y, basic_mel_cfg)

        assert torch.all(torch.isfinite(log_amplitude))

    def test_rea_imag_shapes_match_log_amplitude(self, basic_mel_cfg):
        """rea and imag shapes must match log_amplitude and phase."""
        y = _make_audio(num_samples=24000, batch=1)

        log_amplitude, phase, rea, imag = amplitude_phase_spectrum(y, basic_mel_cfg)

        assert rea.shape == log_amplitude.shape
        assert imag.shape == log_amplitude.shape

    def test_time_dim_positive(self, basic_mel_cfg):
        y = _make_audio(num_samples=24000, batch=1)

        log_amplitude, phase, rea, imag = amplitude_phase_spectrum(y, basic_mel_cfg)

        assert log_amplitude.shape[1] > 0
