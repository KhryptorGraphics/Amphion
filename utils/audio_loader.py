# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Standardized audio loading module using torchaudio backend.

This module provides a unified interface for loading audio files across the codebase,
replacing the inconsistent use of librosa.load(), torchaudio.load(), and whisper.load_audio().

Torchaudio is 2-10x faster than librosa due to its C++ backend, making it the preferred
choice for audio ML training pipelines.
"""

import os
from typing import Optional, Union, Tuple
from pathlib import Path

import torch
import torchaudio


def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True,
    return_sample_rate: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """
    Load audio from a file using torchaudio backend.

    This is the standardized audio loading function for the Amphion codebase.
    It replaces the inconsistent use of librosa.load(), torchaudio.load(), and
    whisper.load_audio() throughout the project.

    Args:
        path: Path to the audio file. Can be a string or Path object.
        sample_rate: Target sample rate for resampling. If None, returns the original
            sample rate without resampling. Common values: 16000, 22050, 24000, 44100, 48000.
        mono: If True, convert multi-channel audio to mono by averaging channels.
            If False, return audio with original channel count. Default: True.
        normalize: If True, normalize audio to the range [-1, 1]. Default: True.
        return_sample_rate: If True, return (waveform, sample_rate). If False,
            return only waveform. Default: False.

    Returns:
        If return_sample_rate is False:
            waveform: torch.Tensor of shape [1, T] if mono=True, [C, T] otherwise.
                T is the number of samples, C is the number of channels.
        If return_sample_rate is True:
            Tuple[waveform, sample_rate] where waveform is as above and sample_rate is int.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio contains NaN or Inf values after loading.

    Example:
        >>> # Load audio at original sample rate
        >>> waveform = load_audio("audio.wav")
        >>> print(waveform.shape)  # [1, T]

        >>> # Load audio at specific sample rate
        >>> waveform, sr = load_audio("audio.wav", sample_rate=24000, return_sample_rate=True)
        >>> print(waveform.shape)  # [1, T]
        >>> print(sr)  # 24000

        >>> # Load stereo audio
        >>> waveform = load_audio("stereo.wav", mono=False)
        >>> print(waveform.shape)  # [2, T]
    """
    # Convert Path to string if necessary
    path = str(path) if isinstance(path, Path) else path

    # Check file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Load audio with torchaudio (C++ backend, fast)
    waveform, sr = torchaudio.load(path)

    # Convert to mono if requested
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Normalize if requested (torchaudio already returns float32 in [-1, 1] for most formats)
    if normalize:
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

    # Resample if target sample rate is specified
    if sample_rate is not None and sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        sr = sample_rate

    # Check for NaN or Inf values
    if torch.isnan(waveform).any() or torch.isinf(waveform).any():
        raise ValueError(f"Audio contains NaN or Inf values: {path}")

    if return_sample_rate:
        return waveform, sr
    return waveform


def load_audio_segment(
    path: Union[str, Path],
    start_time: float,
    duration: float,
    sample_rate: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Load a specific segment from an audio file.

    This function is optimized for loading only a portion of an audio file,
    useful for streaming applications or working with long audio files.

    Args:
        path: Path to the audio file.
        start_time: Start time in seconds.
        duration: Duration to load in seconds.
        sample_rate: Target sample rate. If None, uses original sample rate.
        mono: If True, convert to mono. Default: True.
        normalize: If True, normalize audio. Default: True.

    Returns:
        waveform: torch.Tensor of shape [1, T] if mono=True, [C, T] otherwise.

    Example:
        >>> # Load 5 seconds starting at 10 seconds
        >>> segment = load_audio_segment("long.wav", start_time=10.0, duration=5.0)
        >>> print(segment.shape)  # [1, sample_rate * 5]
    """
    # Load full audio
    waveform, sr = load_audio(
        path, sample_rate=sample_rate, mono=mono, normalize=normalize, return_sample_rate=True
    )

    # Calculate sample positions
    start_sample = int(start_time * sr)
    end_sample = start_sample + int(duration * sr)

    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(waveform.shape[-1], end_sample)

    # Extract segment
    return waveform[:, start_sample:end_sample]


def get_audio_info(path: Union[str, Path]) -> dict:
    """
    Get metadata information about an audio file without loading it.

    Args:
        path: Path to the audio file.

    Returns:
        dict with keys:
            - sample_rate: int, sample rate in Hz
            - num_channels: int, number of audio channels
            - num_frames: int, total number of samples per channel
            - duration: float, duration in seconds
            - format: str, audio format (e.g., 'wav', 'flac', 'mp3')

    Example:
        >>> info = get_audio_info("audio.wav")
        >>> print(f"Duration: {info['duration']:.2f}s, Sample rate: {info['sample_rate']}Hz")
    """
    path = str(path) if isinstance(path, Path) else path

    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Get metadata using torchaudio.info (doesn't load audio data)
    info = torchaudio.info(path)

    return {
        "sample_rate": info.sample_rate,
        "num_channels": info.num_channels,
        "num_frames": info.num_frames,
        "duration": info.num_frames / info.sample_rate if info.sample_rate > 0 else 0,
        "format": os.path.splitext(path)[1].lstrip(".").lower(),
    }