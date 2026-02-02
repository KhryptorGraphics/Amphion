"""
Voice Conversion Routes

Endpoints for voice conversion model inference.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
import logging
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


@router.post("/vevo-voice")
async def vevo_voice_conversion(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="Source audio to convert"),
    reference_audio: UploadFile = File(..., description="Reference voice audio")
):
    """
    Convert voice using Vevo Voice model (full conversion).

    Converts both style and timbre from the source to match the reference.

    Args:
        source_audio: Audio file to convert
        reference_audio: Target voice reference

    Returns:
        FileResponse: Converted audio file
    """
    manager = ModelManager()

    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_src:
        content = await source_audio.read()
        tmp_src.write(content)
        tmp_src_path = tmp_src.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        logger.info("Vevo Voice conversion request")

        sample_rate, audio_data = manager.vevo_vc_inference(
            src_wav=tmp_src_path,
            style_ref_wav=tmp_ref_path,
            timbre_ref_wav=tmp_ref_path  # Same reference for full conversion
        )

        # Save output
        output_path = f"/home/kp/repo2/Amphion/output/web/vevo_voice_{os.urandom(8).hex()}.wav"
        sf.write(output_path, audio_data, sample_rate)

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_src_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="vevo_voice_output.wav"
        )

    except Exception as e:
        cleanup_file(tmp_src_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"Vevo Voice conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vevo-timbre")
async def vevo_timbre_conversion(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="Source audio to convert"),
    reference_audio: UploadFile = File(..., description="Timbre reference audio")
):
    """
    Convert timbre using Vevo Timbre model.

    Preserves the speaking style/prosody while changing the voice timbre.

    Args:
        source_audio: Audio file to convert
        reference_audio: Target timbre reference

    Returns:
        FileResponse: Converted audio file
    """
    manager = ModelManager()

    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_src:
        content = await source_audio.read()
        tmp_src.write(content)
        tmp_src_path = tmp_src.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        logger.info("Vevo Timbre conversion request")

        sample_rate, audio_data = manager.vevo_timbre_inference(
            src_wav=tmp_src_path,
            timbre_ref_wav=tmp_ref_path
        )

        # Save output
        output_path = f"/home/kp/repo2/Amphion/output/web/vevo_timbre_{os.urandom(8).hex()}.wav"
        sf.write(output_path, audio_data, sample_rate)

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_src_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="vevo_timbre_output.wav"
        )

    except Exception as e:
        cleanup_file(tmp_src_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"Vevo Timbre conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vevo-style")
async def vevo_style_conversion(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="Source audio to convert"),
    reference_audio: UploadFile = File(..., description="Style reference audio")
):
    """
    Convert style/accent using Vevo Style model.

    Converts the speaking style/accent while preserving the original timbre.

    Args:
        source_audio: Audio file to convert
        reference_audio: Target style reference

    Returns:
        FileResponse: Converted audio file
    """
    manager = ModelManager()

    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_src:
        content = await source_audio.read()
        tmp_src.write(content)
        tmp_src_path = tmp_src.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        logger.info("Vevo Style conversion request")

        sample_rate, audio_data = manager.vevo_style_inference(
            src_wav=tmp_src_path,
            style_ref_wav=tmp_ref_path
        )

        # Save output
        output_path = f"/home/kp/repo2/Amphion/output/web/vevo_style_{os.urandom(8).hex()}.wav"
        sf.write(output_path, audio_data, sample_rate)

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_src_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="vevo_style_output.wav"
        )

    except Exception as e:
        cleanup_file(tmp_src_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"Vevo Style conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/noro")
async def noro_voice_conversion(
    background_tasks: BackgroundTasks,
    source_audio: UploadFile = File(..., description="Source audio to convert"),
    reference_audio: UploadFile = File(..., description="Target voice reference audio"),
    inference_steps: int = Form(200, description="Number of diffusion steps (150-300)"),
    sigma: float = Form(1.2, description="Sigma parameter (0.95-1.5)")
):
    """
    Convert voice using Noro (noise-robust) model.

    Noro is designed for voice conversion with noisy reference audio.
    Uses diffusion-based generation for high-quality output.

    Args:
        source_audio: Audio file to convert
        reference_audio: Target voice reference (can be noisy)
        inference_steps: Number of diffusion steps (150-300 recommended)
        sigma: Sigma parameter for diffusion (0.95-1.5 recommended)

    Returns:
        FileResponse: Converted audio file
    """
    manager = ModelManager()

    # Save uploaded files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_src:
        content = await source_audio.read()
        tmp_src.write(content)
        tmp_src_path = tmp_src.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        logger.info(f"Noro VC request (steps={inference_steps}, sigma={sigma})")

        sample_rate, audio_data = manager.noro_inference(
            source_wav=tmp_src_path,
            reference_wav=tmp_ref_path,
            inference_steps=inference_steps,
            sigma=sigma
        )

        # Save output
        output_path = f"/home/kp/repo2/Amphion/output/web/noro_{os.urandom(8).hex()}.wav"
        sf.write(output_path, audio_data, sample_rate)

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, tmp_src_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="noro_output.wav"
        )

    except FileNotFoundError as e:
        cleanup_file(tmp_src_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"Noro model not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Noro model not available. Checkpoint needs to be downloaded."
        )
    except Exception as e:
        cleanup_file(tmp_src_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"Noro conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
