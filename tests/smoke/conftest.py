# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke test configuration and shared fixtures.

Skip helpers provided (callable by any test file in this directory):
- has_internet(): True if basic internet connectivity is available
- has_hf_access(): True if huggingface.co is reachable over HTTPS

Fixtures provided:
- skip_if_no_internet: Skips the test if no internet connectivity is detected
- skip_if_no_hf_access: Skips the test if HuggingFace Hub is not reachable
- smoke_prompt_wav: A temporary 24 kHz WAV file for use as prompt audio
"""

import socket

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Connectivity helpers (exported for use by test modules)
# ---------------------------------------------------------------------------


def has_internet(timeout: float = 5.0) -> bool:
    """Return True if basic internet connectivity is available.

    Attempts to open a TCP connection to two well-known DNS servers.
    Returns True as soon as either succeeds, False if both fail.
    """
    for host, port in [("8.8.8.8", 53), ("1.1.1.1", 53)]:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.error, OSError):
            continue
    return False


def has_hf_access(timeout: float = 5.0) -> bool:
    """Return True if huggingface.co is reachable over HTTPS (port 443)."""
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=timeout):
            return True
    except (socket.error, OSError):
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skip_if_no_internet():
    """Skip the current test if no internet connectivity is detected."""
    if not has_internet():
        pytest.skip("No internet connectivity — smoke test skipped")


@pytest.fixture
def skip_if_no_hf_access():
    """Skip the current test if HuggingFace Hub (huggingface.co:443) is not reachable."""
    if not has_hf_access():
        pytest.skip("HuggingFace Hub is not reachable — smoke test skipped")


@pytest.fixture
def smoke_prompt_wav(tmp_path):
    """Create a temporary 24 kHz WAV file containing a synthetic 440 Hz sine wave.

    The 2-second clip is suitable for use as the *prompt_speech_path* argument
    passed to MaskGCT and other TTS inference pipelines.  The file is written
    at 24 000 Hz; librosa will resample it internally when the pipeline loads
    it at a different sample rate.

    Returns:
        str: Absolute path to the created WAV file.
    """
    sample_rate = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.3).astype(np.float32)

    wav_path = tmp_path / "smoke_prompt.wav"
    sf.write(str(wav_path), audio, sample_rate)
    return str(wav_path)
