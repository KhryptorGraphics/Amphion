# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for utils/audio.py

Covers:
- load_audio_torch: load a real temp WAV file, verify returns torch.Tensor and sample rate
- _stft: test on numpy array, verify complex output shape
- energy: test on numpy array, verify 1D output
- get_energy_from_tacotron: test mel and energy shapes using TacotronSTFT
"""

import numpy as np
import pytest
import soundfile as sf
import torch

from utils.audio import _stft, energy, get_energy_from_tacotron, load_audio_torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_file(tmp_path, sample_rate=22050, duration=1.0, freq=440.0):
    """Create a temporary WAV file with a sine wave and return its path."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


def _make_numpy_audio(num_samples=22050, seed=42):
    """Return a float32 numpy array simulating mono audio."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(num_samples).astype(np.float32) * 0.1


# ---------------------------------------------------------------------------
# load_audio_torch
# ---------------------------------------------------------------------------


class TestLoadAudioTorch:
    def test_returns_tensor_and_sample_rate(self, tmp_path):
        """load_audio_torch should return a FloatTensor and the requested sample rate."""
        wav_path = _make_wav_file(tmp_path, sample_rate=22050)
        audio, fs = load_audio_torch(wav_path, 22050)

        assert isinstance(audio, torch.Tensor), "Audio must be a torch.Tensor"
        assert fs == 22050

    def test_uses_synthetic_audio_file_fixture(self, synthetic_audio_file):
        """load_audio_torch works with the shared synthetic_audio_file fixture."""
        audio, fs = load_audio_torch(synthetic_audio_file, 24000)

        assert isinstance(audio, torch.Tensor)
        assert fs == 24000
        assert audio.ndim == 1, "Audio should be 1D (mono)"

    def test_audio_is_float32(self, tmp_path):
        """load_audio_torch should return a float32 tensor."""
        wav_path = _make_wav_file(tmp_path, sample_rate=16000)
        audio, _ = load_audio_torch(wav_path, 16000)

        assert audio.dtype == torch.float32

    def test_audio_is_non_empty(self, tmp_path):
        """Returned audio tensor should have at least one sample."""
        wav_path = _make_wav_file(tmp_path, sample_rate=22050, duration=0.5)
        audio, _ = load_audio_torch(wav_path, 22050)

        assert audio.numel() > 0

    def test_audio_normalized_to_unit_range(self, tmp_path):
        """Audio tensor values should be bounded to roughly [-1, 1]."""
        wav_path = _make_wav_file(tmp_path, sample_rate=22050)
        audio, _ = load_audio_torch(wav_path, 22050)

        assert audio.max().item() <= 1.0 + 1e-5
        assert audio.min().item() >= -1.0 - 1e-5

    def test_resampling_returns_correct_fs(self, tmp_path):
        """When sample rates differ, audio is resampled and returned fs matches request."""
        wav_path = _make_wav_file(tmp_path, sample_rate=44100)
        audio, fs = load_audio_torch(wav_path, 22050)

        assert fs == 22050
        assert isinstance(audio, torch.Tensor)


# ---------------------------------------------------------------------------
# _stft
# ---------------------------------------------------------------------------


class TestStft:
    def test_returns_complex_array(self, basic_mel_cfg):
        """_stft should return a complex numpy array."""
        wav = _make_numpy_audio(num_samples=24000)
        D = _stft(wav, basic_mel_cfg)

        assert isinstance(D, np.ndarray), "Output must be a numpy array"
        assert np.iscomplexobj(D), "Output must be complex"

    def test_output_shape_freq_dim(self, basic_mel_cfg):
        """Frequency dimension should equal n_fft // 2 + 1."""
        wav = _make_numpy_audio(num_samples=24000)
        D = _stft(wav, basic_mel_cfg)

        expected_freq_bins = basic_mel_cfg.n_fft // 2 + 1  # 513
        assert D.shape[0] == expected_freq_bins, (
            f"Expected {expected_freq_bins} freq bins, got {D.shape[0]}"
        )

    def test_output_shape_time_dim_positive(self, basic_mel_cfg):
        """Time dimension must be positive."""
        wav = _make_numpy_audio(num_samples=24000)
        D = _stft(wav, basic_mel_cfg)

        assert D.shape[1] > 0

    def test_output_is_2d(self, basic_mel_cfg):
        """STFT output shape must be 2D: [freq_bins, time_frames]."""
        wav = _make_numpy_audio(num_samples=24000)
        D = _stft(wav, basic_mel_cfg)

        assert D.ndim == 2, f"Expected 2D output, got {D.ndim}D"


# ---------------------------------------------------------------------------
# energy
# ---------------------------------------------------------------------------


class TestEnergy:
    def test_returns_1d_array(self, basic_mel_cfg):
        """energy() should return a 1D numpy array."""
        wav = _make_numpy_audio(num_samples=24000)
        e = energy(wav, basic_mel_cfg)

        assert isinstance(e, np.ndarray), "Energy must be a numpy array"
        assert e.ndim == 1, f"Expected 1D output, got {e.ndim}D"

    def test_output_nonnegative(self, basic_mel_cfg):
        """Energy values (L2 norms of spectrogram rows) must be non-negative."""
        wav = _make_numpy_audio(num_samples=24000)
        e = energy(wav, basic_mel_cfg)

        assert np.all(e >= 0.0), "Energy values must be non-negative"

    def test_output_finite(self, basic_mel_cfg):
        """Energy values must all be finite."""
        wav = _make_numpy_audio(num_samples=24000)
        e = energy(wav, basic_mel_cfg)

        assert np.all(np.isfinite(e)), "Energy values must be finite"

    def test_length_matches_stft_time_frames(self, basic_mel_cfg):
        """Length of energy array must equal the STFT time frame count."""
        wav = _make_numpy_audio(num_samples=24000)
        D = _stft(wav, basic_mel_cfg)
        e = energy(wav, basic_mel_cfg)

        assert e.shape[0] == D.shape[1], (
            f"Expected {D.shape[1]} time frames, got {e.shape[0]}"
        )


# ---------------------------------------------------------------------------
# get_energy_from_tacotron
# ---------------------------------------------------------------------------


class TestGetEnergyFromTacotron:
    """Tests for get_energy_from_tacotron using TacotronSTFT.

    TacotronSTFT.mel_spectrogram calls STFT.transform which unconditionally
    calls .cuda(), so all tests in this class are skipped when CUDA is absent.
    """

    @pytest.fixture
    def tacotron_stft(self):
        """Create a TacotronSTFT instance.

        Skips the test when:
        - CUDA is unavailable (STFT.transform calls .cuda() unconditionally), or
        - TacotronSTFT cannot be instantiated due to librosa API incompatibility.
        """
        if not torch.cuda.is_available():
            pytest.skip("TacotronSTFT requires CUDA (STFT.transform calls .cuda())")

        from utils.stft import TacotronSTFT

        try:
            stft = TacotronSTFT(
                filter_length=1024,
                hop_length=256,
                win_length=1024,
                n_mel_channels=80,
                sampling_rate=22050,
                mel_fmin=0.0,
                mel_fmax=8000.0,
            )
        except Exception as exc:
            pytest.skip(f"TacotronSTFT instantiation failed: {exc}")

        return stft

    def test_returns_mel_and_energy_tuple(self, tacotron_stft):
        """get_energy_from_tacotron should return a (mel, energy) pair."""
        audio = _make_numpy_audio(num_samples=22050)
        result = get_energy_from_tacotron(audio, tacotron_stft)

        assert len(result) == 2

    def test_energy_is_1d_float32_numpy(self, tacotron_stft):
        """Energy from tacotron must be a 1D float32 numpy array."""
        audio = _make_numpy_audio(num_samples=22050)
        _, e = get_energy_from_tacotron(audio, tacotron_stft)

        assert isinstance(e, np.ndarray), "Energy must be a numpy array"
        assert e.ndim == 1, f"Energy must be 1D, got {e.ndim}D"
        assert e.dtype == np.float32, f"Energy dtype must be float32, got {e.dtype}"

    def test_mel_n_mel_channels_dimension(self, tacotron_stft):
        """Mel output should have 80 mel channels on the frequency axis."""
        audio = _make_numpy_audio(num_samples=22050)
        mel, _ = get_energy_from_tacotron(audio, tacotron_stft)

        n_mel_channels = 80
        assert mel.shape[1] == n_mel_channels, (
            f"Expected n_mel_channels={n_mel_channels}, got {mel.shape[1]}"
        )

    def test_energy_nonnegative(self, tacotron_stft):
        """Tacotron energy values (L2 norms) must be non-negative."""
        audio = _make_numpy_audio(num_samples=22050)
        _, e = get_energy_from_tacotron(audio, tacotron_stft)

        assert np.all(e >= 0.0), "Tacotron energy values must be non-negative"

    def test_energy_length_matches_mel_time_dim(self, tacotron_stft):
        """Energy length must match the time dimension of the mel spectrogram."""
        audio = _make_numpy_audio(num_samples=22050)
        mel, e = get_energy_from_tacotron(audio, tacotron_stft)

        mel_time_dim = mel.shape[-1]
        assert e.shape[0] == mel_time_dim, (
            f"Energy length {e.shape[0]} must match mel time dim {mel_time_dim}"
        )
