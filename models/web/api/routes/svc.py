"""
SVC Routes

Endpoints for singing voice conversion model inference.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
import logging
from typing import Optional
import soundfile as sf
import numpy as np
import torch

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


@router.post("/vevosing")
async def vevosing_svc(
    background_tasks: BackgroundTasks,
    content_audio: UploadFile = File(..., description="Source audio (content/melody)"),
    reference_audio: UploadFile = File(..., description="Reference audio (timbre)"),
    mode: str = Form("fm", description="Inference mode: 'fm' for timbre-only, 'ar' for full control"),
    use_shifted_src: bool = Form(True, description="Use pitch-shifted source for prosody extraction"),
    flow_matching_steps: int = Form(32, description="Number of flow matching steps"),
):
    """
    Singing Voice Conversion using VevoSing (Vevo1.5).

    Converts the singing voice in content_audio to match the timbre of reference_audio.

    Args:
        content_audio: Source audio file containing the singing voice to convert
        reference_audio: Reference audio file for target timbre
        mode: 'fm' for flow-matching only (timbre control), 'ar' for AR+FM (full control)
        use_shifted_src: Use pitch-shifted source for prosody extraction
        flow_matching_steps: Number of flow matching steps (higher = better quality, slower)

    Returns:
        FileResponse: Converted audio file
    """
    manager = ModelManager()

    # Save uploaded audio files to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_content:
        content = await content_audio.read()
        tmp_content.write(content)
        tmp_content_path = tmp_content.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        logger.info(f"VevoSing SVC request: mode={mode}, steps={flow_matching_steps}")

        sample_rate, audio_data = manager.vevosing_inference(
            content_wav_path=tmp_content_path,
            reference_wav_path=tmp_ref_path,
            mode=mode,
            use_shifted_src=use_shifted_src,
            flow_matching_steps=flow_matching_steps,
        )

        # Save output
        output_path = f"/home/kp/repo2/Amphion/output/web/vevosing_{os.urandom(8).hex()}.wav"

        # Convert torch tensor to numpy if needed
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()
        # Squeeze batch dimension if present [1, T] -> [T]
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
        # Ensure float32 format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        sf.write(output_path, audio_data, sample_rate)

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_content_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="vevosing_output.wav"
        )

    except Exception as e:
        cleanup_file(tmp_content_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"VevoSing SVC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
