"""
Vocoder Routes

Endpoints for neural vocoder inference.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
import logging
import numpy as np

from ..models.manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


def cleanup_file(file_path: str):
    """Background task to cleanup temporary files."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up {file_path}: {e}")


@router.post("/hifigan")
async def hifigan_vocoder(
    background_tasks: BackgroundTasks,
    mel: UploadFile = File(..., description="Mel-spectrogram numpy file (.npy)"),
):
    """
    Convert mel-spectrogram to audio using HiFi-GAN vocoder.

    HiFi-GAN is a GAN-based neural vocoder that generates high-fidelity audio
    from mel-spectrogram features.

    Args:
        mel: Numpy file containing mel-spectrogram array [n_mels, time_frames]

    Returns:
        FileResponse: Generated audio file (WAV format, 22.05kHz)
    """
    manager = ModelManager()

    # Save uploaded mel file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_mel:
        content = await mel.read()
        tmp_mel.write(content)
        tmp_mel_path = tmp_mel.name

    try:
        logger.info(f"HiFi-GAN vocoder request")

        output_path = manager.vocoder_inference(
            mel_path=tmp_mel_path,
            vocoder_type="hifigan",
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_mel_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="hifigan_output.wav",
            background=background_tasks
        )

    except Exception as e:
        cleanup_file(tmp_mel_path)
        logger.error(f"HiFi-GAN vocoder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bigvgan")
async def bigvgan_vocoder(
    background_tasks: BackgroundTasks,
    mel: UploadFile = File(..., description="Mel-spectrogram numpy file (.npy)"),
):
    """
    Convert mel-spectrogram to audio using BigVGAN vocoder.

    BigVGAN is a large-scale GAN-based vocoder supporting multiple sampling rates
    (22.05kHz, 44.1kHz) and bandwidths (80-band, 128-band mels).

    Args:
        mel: Numpy file containing mel-spectrogram array [n_mels, time_frames]

    Returns:
        FileResponse: Generated audio file (WAV format, 22.05kHz)
    """
    manager = ModelManager()

    # Save uploaded mel file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_mel:
        content = await mel.read()
        tmp_mel.write(content)
        tmp_mel_path = tmp_mel.name

    try:
        logger.info(f"BigVGAN vocoder request")

        output_path = manager.vocoder_inference(
            mel_path=tmp_mel_path,
            vocoder_type="bigvgan",
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_mel_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="bigvgan_output.wav",
            background=background_tasks
        )

    except Exception as e:
        cleanup_file(tmp_mel_path)
        logger.error(f"BigVGAN vocoder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generic")
async def generic_vocoder(
    background_tasks: BackgroundTasks,
    mel: UploadFile = File(..., description="Mel-spectrogram numpy file (.npy)"),
    vocoder_type: str = Form("hifigan", description="Vocoder type: 'hifigan' or 'bigvgan'"),
):
    """
    Convert mel-spectrogram to audio using specified neural vocoder.

    Universal endpoint supporting multiple vocoder types.

    Args:
        mel: Numpy file containing mel-spectrogram array [n_mels, time_frames]
        vocoder_type: Type of vocoder to use ("hifigan" or "bigvgan")

    Returns:
        FileResponse: Generated audio file (WAV format)
    """
    manager = ModelManager()

    if vocoder_type not in ["hifigan", "bigvgan"]:
        raise HTTPException(status_code=400, detail=f"Unsupported vocoder type: {vocoder_type}")

    # Save uploaded mel file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_mel:
        content = await mel.read()
        tmp_mel.write(content)
        tmp_mel_path = tmp_mel.name

    try:
        logger.info(f"Generic vocoder request: type={vocoder_type}")

        output_path = manager.vocoder_inference(
            mel_path=tmp_mel_path,
            vocoder_type=vocoder_type,
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_mel_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{vocoder_type}_output.wav",
            background=background_tasks
        )

    except Exception as e:
        cleanup_file(tmp_mel_path)
        logger.error(f"Vocoder error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
