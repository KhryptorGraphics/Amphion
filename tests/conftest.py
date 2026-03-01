# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Root-level pytest conftest.py providing shared fixtures for all test modules.

Fixtures provided:
- synthetic_audio_file: A temporary WAV file with a synthetic sine wave
- basic_mel_cfg: JsonHParams with default mel spectrogram configuration
- basic_preprocess_cfg: JsonHParams with full preprocessing configuration
- cpu_device: torch.device for CPU
- gpu_device: torch.device for GPU (skips if CUDA unavailable)
"""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from utils.util import JsonHParams


@pytest.fixture
def synthetic_audio_file(tmp_path):
    """Create a temporary WAV file containing a synthetic sine wave.

    Returns the path to the created WAV file.
    """
    sample_rate = 24000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)

    wav_path = tmp_path / "synthetic.wav"
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)


@pytest.fixture
def basic_mel_cfg():
    """JsonHParams with default mel spectrogram configuration.

    Matches the mel-related fields from config/base.json.
    """
    return JsonHParams(
        n_mel=80,
        win_size=480,
        hop_size=120,
        sample_rate=24000,
        n_fft=1024,
        fmin=0,
        fmax=12000,
        min_level_db=-115,
        ref_level_db=20,
        bits=8,
    )


@pytest.fixture
def basic_preprocess_cfg():
    """JsonHParams with full preprocessing configuration.

    Matches the preprocess section from config/base.json.
    """
    return JsonHParams(
        # Audio trimming
        phone_extractor="espeak",
        data_augment=False,
        trim_silence=False,
        num_silent_frames=8,
        trim_fft_size=512,
        trim_hop_size=128,
        trim_top_db=30,
        # Acoustic feature extraction flags
        extract_mel=False,
        mel_extract_mode="",
        extract_linear_spec=False,
        extract_mcep=False,
        extract_pitch=False,
        extract_acoustic_token=False,
        pitch_remove_outlier=False,
        extract_uv=False,
        pitch_norm=False,
        extract_audio=False,
        extract_label=False,
        pitch_extractor="parselmouth",
        extract_energy=False,
        energy_remove_outlier=False,
        energy_norm=False,
        energy_extract_mode="from_mel",
        extract_duration=False,
        extract_amplitude_phase=False,
        mel_min_max_norm=False,
        # Linguistic features
        extract_phone=False,
        lexicon_path="./text/lexicon/librispeech-lexicon.txt",
        # Content features
        extract_whisper_feature=False,
        extract_contentvec_feature=False,
        extract_mert_feature=False,
        extract_wenet_feature=False,
        # Mel spectrogram settings
        n_mel=80,
        win_size=480,
        hop_size=120,
        sample_rate=24000,
        n_fft=1024,
        fmin=0,
        fmax=12000,
        min_level_db=-115,
        ref_level_db=20,
        bits=8,
        # Directory names
        processed_dir="processed_data",
        trimmed_wav_dir="trimmed_wavs",
        raw_data="raw_data",
        phone_dir="phones",
        wav_dir="wavs",
        audio_dir="audios",
        mel_dir="mels",
        mcep_dir="mcep",
        dur_dir="durs",
        pitch_dir="pitches",
        energy_dir="energys",
        uv_dir="uvs",
        # Feature flags for training
        use_text=False,
        use_phone=False,
        use_phn_seq=False,
        use_lab=False,
        use_linear=False,
    )


@pytest.fixture
def cpu_device():
    """Return a torch.device for CPU computation."""
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    """Return a torch.device for GPU computation.

    Skips the test automatically if CUDA is not available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this machine")
    return torch.device("cuda")
