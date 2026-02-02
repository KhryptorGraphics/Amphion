"""
TTS Routes

Endpoints for text-to-speech model inference.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
import logging
from typing import Optional
import soundfile as sf

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


@router.post("/maskgct")
async def maskgct_tts(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Reference audio file"),
    text: str = Form(..., description="Text to synthesize"),
    target_language: Optional[str] = Form(None, description="Target language (auto-detected if not provided)"),
    target_len: float = Form(-1, description="Target duration in seconds (-1 for auto)"),
    n_timesteps: int = Form(25, description="Number of diffusion steps")
):
    """
    Generate speech using MaskGCT model.

    Args:
        audio: Reference audio file for voice cloning
        text: Text to synthesize
        target_language: Optional target language override
        target_len: Target duration in seconds (-1 for automatic)
        n_timesteps: Number of diffusion steps

    Returns:
        FileResponse: Generated audio file
    """
    manager = ModelManager()

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        content = await audio.read()
        tmp_audio.write(content)
        tmp_audio_path = tmp_audio.name

    try:
        logger.info(f"MaskGCT inference request: text='{text[:50]}...', target_len={target_len}")

        output_path = manager.maskgct_inference(
            prompt_wav_path=tmp_audio_path,
            target_text=text,
            target_len=target_len,
            n_timesteps=n_timesteps
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_audio_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="maskgct_output.wav",
            background=background_tasks
        )

    except Exception as e:
        cleanup_file(tmp_audio_path)
        logger.error(f"MaskGCT inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dualcodec-valle")
async def dualcodec_valle_tts(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Reference audio file"),
    text: str = Form(..., description="Text to synthesize"),
    ref_text: Optional[str] = Form("", description="Reference audio transcript"),
    temperature: float = Form(1.0, description="Sampling temperature"),
    top_k: int = Form(15, description="Top-K sampling"),
    top_p: float = Form(0.85, description="Top-P sampling"),
    repeat_penalty: float = Form(1.1, description="Repeat penalty")
):
    """
    Generate speech using DualCodec-VALLE model.

    Args:
        audio: Reference audio file
        text: Text to synthesize
        ref_text: Optional transcript of reference audio
        temperature: Sampling temperature
        top_k: Top-K sampling parameter
        top_p: Top-P sampling parameter
        repeat_penalty: Repeat penalty

    Returns:
        FileResponse: Generated audio file
    """
    manager = ModelManager()

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        content = await audio.read()
        tmp_audio.write(content)
        tmp_audio_path = tmp_audio.name

    try:
        logger.info(f"DualCodec-VALLE inference request: text='{text[:50]}...'")

        sample_rate, audio_data = manager.dualcodec_valle_inference(
            ref_audio=tmp_audio_path,
            ref_text=ref_text or "",
            gen_text=text,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )

        # Save output
        output_path = f"/home/kp/repo2/Amphion/output/web/valle_{os.urandom(8).hex()}.wav"
        sf.write(output_path, audio_data, sample_rate)

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_audio_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="valle_output.wav"
        )

    except Exception as e:
        cleanup_file(tmp_audio_path)
        logger.error(f"DualCodec-VALLE inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vevo")
async def vevo_tts(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(..., description="Reference audio file"),
    text: str = Form(..., description="Text to synthesize"),
    ref_text: Optional[str] = Form("", description="Reference audio transcript"),
    timbre_audio: Optional[UploadFile] = File(None, description="Optional timbre reference audio"),
    src_language: str = Form("en", description="Source text language"),
    ref_language: str = Form("en", description="Reference audio language")
):
    """
    Generate speech using Vevo TTS model.

    Args:
        audio: Style reference audio file
        text: Text to synthesize
        ref_text: Optional transcript of reference audio
        timbre_audio: Optional separate timbre reference
        src_language: Source text language
        ref_language: Reference audio language

    Returns:
        FileResponse: Generated audio file
    """
    manager = ModelManager()

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        content = await audio.read()
        tmp_audio.write(content)
        tmp_audio_path = tmp_audio.name

    tmp_timbre_path = None
    if timbre_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_timbre:
            content = await timbre_audio.read()
            tmp_timbre.write(content)
            tmp_timbre_path = tmp_timbre.name

    try:
        logger.info(f"Vevo TTS inference request: text='{text[:50]}...'")

        sample_rate, audio_data = manager.vevo_tts_inference(
            src_text=text,
            ref_wav=tmp_audio_path,
            timbre_ref_wav=tmp_timbre_path,
            ref_text=ref_text if ref_text else None,
            src_language=src_language,
            ref_language=ref_language
        )

        # Save output
        output_path = f"/home/kp/repo2/Amphion/output/web/vevo_tts_{os.urandom(8).hex()}.wav"
        sf.write(output_path, audio_data, sample_rate)

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_audio_path)
        if tmp_timbre_path:
            background_tasks.add_task(cleanup_file, tmp_timbre_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="vevo_tts_output.wav"
        )

    except Exception as e:
        cleanup_file(tmp_audio_path)
        if tmp_timbre_path:
            cleanup_file(tmp_timbre_path)
        logger.error(f"Vevo TTS inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
