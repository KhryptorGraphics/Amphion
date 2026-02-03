"""
TTA Routes

Endpoints for text-to-audio model inference.
"""

from fastapi import APIRouter, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
import logging
from typing import Optional

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


@router.post("/audioldm")
async def audioldm_tta(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="Text description of desired audio"),
    duration: float = Form(10.0, description="Target duration in seconds"),
    num_inference_steps: int = Form(50, description="Number of diffusion steps"),
    guidance_scale: float = Form(3.5, description="Classifier-free guidance scale"),
):
    """
    Generate audio from text description using AudioLDM model.

    AudioLDM uses latent diffusion to generate sound effects, ambient audio,
    and music from text descriptions.

    Args:
        text: Text description of desired audio (e.g., "dog barking in a park")
        duration: Target duration in seconds (default: 10.0)
        num_inference_steps: Number of diffusion steps, higher = better quality (default: 50)
        guidance_scale: Classifier-free guidance scale (default: 3.5)

    Returns:
        FileResponse: Generated audio file (WAV format)
    """
    manager = ModelManager()

    try:
        logger.info(f"AudioLDM TTA request: text='{text[:50]}...', duration={duration}")

        output_path = manager.tta_audioldm_inference(
            text_prompt=text,
            duration=duration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="audioldm_output.wav",
            background=background_tasks
        )

    except Exception as e:
        logger.error(f"AudioLDM inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/picoaudio")
async def picoaudio_tta(
    background_tasks: BackgroundTasks,
    text: str = Form(..., description="Text description of desired audio"),
    duration: float = Form(10.0, description="Target duration in seconds"),
    num_inference_steps: int = Form(25, description="Number of generation steps"),
):
    """
    Generate audio from text description using PicoAudio model.

    PicoAudio is a lightweight text-to-audio model optimized for fast generation
    of sound effects and short audio clips.

    Args:
        text: Text description of desired audio (e.g., "car engine starting")
        duration: Target duration in seconds (default: 10.0)
        num_inference_steps: Number of generation steps (default: 25)

    Returns:
        FileResponse: Generated audio file (WAV format)
    """
    manager = ModelManager()

    try:
        logger.info(f"PicoAudio TTA request: text='{text[:50]}...', duration={duration}")

        output_path = manager.tta_picoaudio_inference(
            text_prompt=text,
            duration=duration,
            num_inference_steps=num_inference_steps,
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="picoaudio_output.wav",
            background=background_tasks
        )

    except Exception as e:
        logger.error(f"PicoAudio inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
