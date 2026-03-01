# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
import torchaudio


@dataclass
class InferenceOutput:
    """Standardized container for a single inference result.

    Attributes:
        uid: Unique identifier for this utterance (used as the output filename stem).
        audio: Waveform as a numpy array of shape [samples] or [1, samples].
        sample_rate: Sample rate of the waveform in Hz.
        text: Optional transcript or input text associated with the audio.
        extra_metadata: Optional dict of model-specific metadata to include in the manifest.
    """

    uid: str
    audio: np.ndarray
    sample_rate: int
    text: Optional[str] = None
    extra_metadata: Optional[dict] = field(default=None)


class InferenceOutputWriter:
    """Writes inference outputs (audio files + manifest.json) to a directory.

    Audio files are saved immediately on each :meth:`add` call so that the
    writer never holds large buffers in memory.  Call :meth:`save_manifest`
    once all outputs have been added to flush the manifest.

    Args:
        output_dir: Directory where audio files and manifest.json will be written.
        model_name: Human-readable model identifier stored in the manifest.
        sample_rate: Default sample rate used when writing audio files.
    """

    def __init__(self, output_dir: str, model_name: str, sample_rate: int):
        self.output_dir = output_dir
        self.model_name = model_name
        self.sample_rate = sample_rate
        self._utterances: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)

    def add(self, output: InferenceOutput) -> str:
        """Save *output* audio to disk and record its metadata for the manifest.

        The audio is written immediately so that callers can discard the numpy
        array after this call.

        Args:
            output: The inference result to persist.

        Returns:
            The relative path (relative to *output_dir*) of the saved wav file.
        """
        audio_filename = output.uid + ".wav"
        audio_path = os.path.join(self.output_dir, audio_filename)

        # Normalise to a 2-D float32 tensor [1, samples] – same approach as
        # utils/io.py save_audio().
        waveform = torch.as_tensor(output.audio, dtype=torch.float32, device="cpu")
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.size(0) != 1:
            # Stereo to mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        torchaudio.save(
            audio_path,
            waveform,
            output.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16,
        )

        duration_seconds = waveform.shape[-1] / output.sample_rate

        utterance: dict = {
            "uid": output.uid,
            "audio_path": audio_filename,
            "duration_seconds": round(duration_seconds, 6),
        }
        if output.text is not None:
            utterance["text"] = output.text
        if output.extra_metadata is not None:
            utterance["extra_metadata"] = output.extra_metadata

        self._utterances.append(utterance)
        return audio_filename

    def save_manifest(self) -> str:
        """Write a ``manifest.json`` file summarising all added outputs.

        Returns:
            The absolute path of the written manifest file.
        """
        manifest = {
            "model": self.model_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_rate": self.sample_rate,
            "utterances": self._utterances,
        }
        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return manifest_path
