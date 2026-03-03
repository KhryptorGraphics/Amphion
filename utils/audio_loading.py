# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unified audio loading utility backed by torchaudio.

Provides a drop-in replacement for ``librosa.load`` with an optional
thread-safe LRU memory cache.  The torchaudio C++ backend is 2-10x
faster than librosa for typical audio files.

Usage::

    from utils.audio_loading import load_audio, load_audio_tensor

    # Same return contract as librosa.load(path, sr=…)
    wav_np, sr = load_audio(path, sample_rate=24000)

    # Torch tensor variant (shape [sequence_len], float32)
    wav_t, sr = load_audio_tensor(path, sample_rate=16000)
"""

import threading
from collections import OrderedDict

import numpy as np
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Thread-safe LRU cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()
_cache_max_size: int = 1000


def _cache_get(key):
    """Return cached value or ``None`` if not present."""
    with _cache_lock:
        if key not in _cache:
            return None
        # Move to end (most-recently used)
        _cache.move_to_end(key)
        return _cache[key]


def _cache_put(key, value, max_size: int):
    """Insert value; evict oldest entry if cache is full."""
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
        _cache[key] = value
        while len(_cache) > max_size:
            _cache.popitem(last=False)


def clear_audio_cache():
    """Remove all entries from the in-process audio cache."""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Core loading helpers
# ---------------------------------------------------------------------------


def _load_waveform_tensor(path, sample_rate=None, mono=True):
    """Load audio with torchaudio and return a float32 Tensor of shape ``[T]``.

    Args:
        path: Path to the audio file (str, bytes, or path-like).
        sample_rate (int | None): Target sample rate.  If ``None`` the file's
            native rate is kept.
        mono (bool): Downmix to mono when ``True``.

    Returns:
        waveform (torch.Tensor): Float32 1-D tensor of shape ``[T]``.
        sr (int): Sample rate of the returned waveform.
    """
    waveform, file_sr = torchaudio.load(path)

    # Downmix to mono: shape [C, T] -> [1, T] -> [T]
    if mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if a target rate is specified and differs from file rate
    if sample_rate is not None and file_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, file_sr, sample_rate)
        out_sr = sample_rate
    else:
        out_sr = file_sr

    # Flatten channel dim: [1, T] -> [T]  (or [T] already if not mono-mixed above)
    waveform = waveform.squeeze(0)

    return waveform.to(dtype=torch.float32), int(out_sr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_audio(path, sample_rate=None, mono=True, use_cache=False, cache_max_size=1000):
    """Load an audio file and return a NumPy float32 array.

    Drop-in replacement for ``librosa.load(path, sr=sample_rate, mono=mono)``.

    Args:
        path: Path to the audio file.
        sample_rate (int | None): Target sample rate.  When ``None`` the file's
            native rate is returned unchanged.
        mono (bool): Downmix multi-channel audio to mono.  Defaults to ``True``.
        use_cache (bool): Cache the result in an in-process LRU dict so that
            repeated calls for the same ``(path, sample_rate, mono)`` triple
            return immediately.  The cache is shared across all callers and is
            thread-safe.  Defaults to ``False``.
        cache_max_size (int): Maximum number of entries in the cache.  When the
            cache is full the oldest entry is evicted.  Defaults to 1000.

    Returns:
        waveform (np.ndarray): Float32 array of shape ``[T]``.
        sample_rate (int): Sample rate of the returned waveform.
    """
    cache_key = (str(path), sample_rate, mono)

    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    waveform_t, out_sr = _load_waveform_tensor(path, sample_rate=sample_rate, mono=mono)
    result = (waveform_t.numpy(), out_sr)

    if use_cache:
        _cache_put(cache_key, result, max_size=cache_max_size)

    return result


def load_audio_tensor(
    path, sample_rate=None, mono=True, use_cache=False, cache_max_size=1000
):
    """Load an audio file and return a ``torch.Tensor``.

    Same behaviour as :func:`load_audio` but returns a ``torch.float32``
    tensor instead of a NumPy array, avoiding an extra copy when the caller
    immediately converts to a tensor anyway.

    Args:
        path: Path to the audio file.
        sample_rate (int | None): Target sample rate.
        mono (bool): Downmix to mono.  Defaults to ``True``.
        use_cache (bool): Enable in-process LRU caching.
        cache_max_size (int): Maximum cache entries.

    Returns:
        waveform (torch.Tensor): Float32 tensor of shape ``[T]``.
        sample_rate (int): Sample rate of the returned waveform.
    """
    cache_key = ("tensor", str(path), sample_rate, mono)

    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    waveform_t, out_sr = _load_waveform_tensor(path, sample_rate=sample_rate, mono=mono)
    result = (waveform_t, out_sr)

    if use_cache:
        _cache_put(cache_key, result, max_size=cache_max_size)

    return result
