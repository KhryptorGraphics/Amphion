"""
Demucs Source Separation

Integrates Meta's htdemucs model for separating vocals from accompaniment.
Used by the SVC pipeline's "Full Song Mode" to process complete songs.
"""

import torch
import torchaudio
import numpy as np
import soundfile as sf
import tempfile
import logging
import os

logger = logging.getLogger(__name__)


class DemucsSourceSeparator:
    """Lazy-loaded Demucs source separator for extracting vocals."""

    def __init__(self, device):
        self.device = device
        self._loaded = False
        self._model = None

    def load(self):
        """Load the htdemucs pretrained model."""
        if self._loaded:
            return

        logger.info("Loading Demucs htdemucs model...")
        from demucs.pretrained import get_model

        self._model = get_model("htdemucs")
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        logger.info("Demucs model loaded successfully")

    def separate(self, audio_path: str) -> tuple:
        """
        Separate vocals from accompaniment.

        Args:
            audio_path: Path to input audio file

        Returns:
            (vocals_path, accompaniment_path) as temp WAV files at 44100Hz
        """
        self.load()
        from demucs.apply import apply_model

        # Load audio
        wav, sr = torchaudio.load(audio_path)

        # Resample to 44100Hz if needed (Demucs expects 44.1kHz)
        if sr != 44100:
            logger.info(f"Resampling from {sr}Hz to 44100Hz for Demucs")
            resampler = torchaudio.transforms.Resample(sr, 44100)
            wav = resampler(wav)

        # Ensure stereo (Demucs expects 2 channels)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]

        # Add batch dimension [B, C, T]
        wav = wav.unsqueeze(0).to(self.device)

        # Run separation
        logger.info("Running Demucs source separation...")
        with torch.no_grad():
            sources = apply_model(self._model, wav)

        # sources shape: [B, num_sources, C, T]
        # htdemucs source order: drums, bass, other, vocals
        source_names = self._model.sources
        vocals_idx = source_names.index("vocals")

        vocals = sources[0, vocals_idx]  # [C, T]

        # Sum non-vocal stems into accompaniment
        accompaniment = torch.zeros_like(vocals)
        for i, name in enumerate(source_names):
            if name != "vocals":
                accompaniment += sources[0, i]

        # Save to temp files
        vocals_path = tempfile.mktemp(suffix="_vocals.wav")
        accompaniment_path = tempfile.mktemp(suffix="_accompaniment.wav")

        torchaudio.save(vocals_path, vocals.cpu(), 44100)
        torchaudio.save(accompaniment_path, accompaniment.cpu(), 44100)

        logger.info(f"Separation complete: vocals={vocals_path}, accompaniment={accompaniment_path}")
        return vocals_path, accompaniment_path

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Demucs model unloaded")


def remix_audio(
    accompaniment_path: str,
    original_vocals_path: str,
    converted_vocals_path: str,
    output_path: str,
    vocals_volume_db: float = 0.0,
):
    """
    Remix converted vocals with original accompaniment.

    Resamples converted vocals (typically 24kHz mono from SVC) to 44.1kHz stereo,
    peak-matches to original vocals level, applies optional gain, and mixes.

    Args:
        accompaniment_path: Path to accompaniment WAV (44.1kHz stereo from Demucs)
        original_vocals_path: Path to original vocals WAV (44.1kHz stereo, for level reference)
        converted_vocals_path: Path to converted vocals WAV (from SVC, any sample rate)
        output_path: Path to write the final mixed WAV
        vocals_volume_db: Additional gain adjustment in dB (default 0.0)
    """
    import librosa

    # Load accompaniment (44.1kHz stereo)
    accomp, accomp_sr = sf.read(accompaniment_path, always_2d=True)

    # Load original vocals for level reference
    orig_vocals, orig_sr = sf.read(original_vocals_path, always_2d=True)

    # Load converted vocals
    conv_vocals, conv_sr = sf.read(converted_vocals_path, always_2d=True)

    # Resample converted vocals to 44100Hz if needed
    if conv_sr != 44100:
        logger.info(f"Resampling converted vocals from {conv_sr}Hz to 44100Hz")
        # librosa expects [channels, samples] for multi-channel
        conv_vocals_t = conv_vocals.T
        resampled = []
        for ch in range(conv_vocals_t.shape[0]):
            resampled.append(
                librosa.resample(conv_vocals_t[ch], orig_sr=conv_sr, target_sr=44100)
            )
        conv_vocals = np.array(resampled).T

    # Expand mono to stereo if needed
    if conv_vocals.ndim == 1:
        conv_vocals = np.column_stack([conv_vocals, conv_vocals])
    elif conv_vocals.shape[1] == 1:
        conv_vocals = np.column_stack([conv_vocals[:, 0], conv_vocals[:, 0]])

    # Peak-match converted vocals to original vocals level
    orig_peak = np.max(np.abs(orig_vocals)) + 1e-8
    conv_peak = np.max(np.abs(conv_vocals)) + 1e-8
    gain = orig_peak / conv_peak
    conv_vocals = conv_vocals * gain

    # Apply optional volume adjustment
    if vocals_volume_db != 0.0:
        linear_gain = 10 ** (vocals_volume_db / 20.0)
        conv_vocals = conv_vocals * linear_gain

    # Align lengths (pad shorter or trim longer)
    target_len = accomp.shape[0]
    if conv_vocals.shape[0] < target_len:
        pad = np.zeros((target_len - conv_vocals.shape[0], conv_vocals.shape[1]))
        conv_vocals = np.concatenate([conv_vocals, pad], axis=0)
    elif conv_vocals.shape[0] > target_len:
        conv_vocals = conv_vocals[:target_len]

    # Mix
    mixed = accomp + conv_vocals

    # Peak-limit at 0.99 to prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)

    # Write output
    sf.write(output_path, mixed, 44100)
    logger.info(f"Remix complete: {output_path}")
