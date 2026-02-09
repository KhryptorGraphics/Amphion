"""
SVC Routes

Endpoints for singing voice conversion model inference.
Supports Full Song Mode: Demucs source separation -> SVC -> remix.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
import logging
from typing import Callable, Optional, Tuple
import soundfile as sf
import numpy as np
import torch

from ..models.manager import ModelManager
from ..upload_validation import validate_audio_file

logger = logging.getLogger(__name__)
router = APIRouter()

OUTPUT_DIR = "/home/kp/repo2/Amphion/output/web"


def cleanup_file(file_path: str):
    """Background task to cleanup temporary files."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up {file_path}: {e}")


def _save_svc_output(audio_data, sample_rate: int, prefix: str) -> str:
    """Normalize SVC output and save to WAV file."""
    if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()
    if audio_data.ndim > 1:
        audio_data = audio_data.squeeze()
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    output_path = f"{OUTPUT_DIR}/{prefix}_{os.urandom(8).hex()}.wav"
    sf.write(output_path, audio_data, sample_rate)
    return output_path


def _run_full_song_pipeline(
    manager: ModelManager,
    content_path: str,
    reference_path: str,
    svc_inference_fn: Callable,
    svc_kwargs: dict,
    output_prefix: str,
    background_tasks: BackgroundTasks,
    vocals_volume_db: float = 0.0,
) -> FileResponse:
    """
    Full Song Mode pipeline: Demucs separate -> SVC on vocals -> remix.

    Args:
        manager: ModelManager instance
        content_path: Path to uploaded content audio (full song)
        reference_path: Path to uploaded reference audio
        svc_inference_fn: The SVC model's inference method
        svc_kwargs: Additional kwargs for the SVC inference (excluding content/reference paths)
        output_prefix: Prefix for output filename
        background_tasks: FastAPI background tasks for cleanup
        vocals_volume_db: Vocal volume adjustment in dB

    Returns:
        FileResponse with the remixed audio
    """
    from ..models.demucs_separator import remix_audio

    # Step 1: Separate vocals from accompaniment
    logger.info("Full Song Mode: Separating vocals with Demucs...")
    vocals_path, accompaniment_path = manager.demucs_separate(content_path)

    try:
        # Step 2: Run SVC on extracted vocals
        logger.info("Full Song Mode: Running SVC on extracted vocals...")
        sample_rate, audio_data = svc_inference_fn(
            content_wav_path=vocals_path,
            reference_wav_path=reference_path,
            **svc_kwargs,
        )

        # Save SVC output to temp file
        converted_vocals_path = _save_svc_output(audio_data, sample_rate, f"{output_prefix}_converted_vocals")

        # Step 3: Remix converted vocals with accompaniment
        logger.info("Full Song Mode: Remixing with accompaniment...")
        final_output_path = f"{OUTPUT_DIR}/{output_prefix}_fullsong_{os.urandom(8).hex()}.wav"
        remix_audio(
            accompaniment_path=accompaniment_path,
            original_vocals_path=vocals_path,
            converted_vocals_path=converted_vocals_path,
            output_path=final_output_path,
            vocals_volume_db=vocals_volume_db,
        )

        # Schedule cleanup of all temp files
        background_tasks.add_task(cleanup_file, vocals_path)
        background_tasks.add_task(cleanup_file, accompaniment_path)
        background_tasks.add_task(cleanup_file, converted_vocals_path)
        background_tasks.add_task(cleanup_file, final_output_path)

        return FileResponse(
            final_output_path,
            media_type="audio/wav",
            filename=f"{output_prefix}_output.wav",
        )

    except Exception:
        # Clean up separation files on error
        cleanup_file(vocals_path)
        cleanup_file(accompaniment_path)
        raise


@router.post("/vevosing")
async def vevosing_svc(
    background_tasks: BackgroundTasks,
    content_audio: UploadFile = File(..., description="Source audio (content/melody)"),
    reference_audio: UploadFile = File(..., description="Reference audio (timbre)"),
    mode: str = Form("fm", description="Inference mode: 'fm' for timbre-only, 'ar' for full control"),
    use_shifted_src: bool = Form(True, description="Use pitch-shifted source for prosody extraction"),
    flow_matching_steps: int = Form(32, description="Number of flow matching steps"),
    full_song_mode: bool = Form(False, description="Enable Full Song Mode (auto-separate vocals, convert, remix)"),
    vocals_volume_db: float = Form(0.0, description="Vocal volume adjustment in dB (-6 to +6)"),
):
    """
    Singing Voice Conversion using VevoSing (Vevo1.5).

    Converts the singing voice in content_audio to match the timbre of reference_audio.
    When full_song_mode is enabled, automatically separates vocals from instrumentals,
    converts the vocals, and remixes with the original accompaniment.
    """
    manager = ModelManager()

    content_audio = await validate_audio_file(content_audio, "content_audio", max_size=100*1024*1024)
    reference_audio = await validate_audio_file(reference_audio, "reference_audio", max_size=50*1024*1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_content:
        content = await content_audio.read()
        tmp_content.write(content)
        tmp_content_path = tmp_content.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        svc_kwargs = dict(
            mode=mode,
            use_shifted_src=use_shifted_src,
            flow_matching_steps=flow_matching_steps,
        )

        if full_song_mode:
            logger.info(f"VevoSing Full Song Mode: mode={mode}, steps={flow_matching_steps}")
            response = _run_full_song_pipeline(
                manager=manager,
                content_path=tmp_content_path,
                reference_path=tmp_ref_path,
                svc_inference_fn=manager.vevosing_inference,
                svc_kwargs=svc_kwargs,
                output_prefix="vevosing",
                background_tasks=background_tasks,
                vocals_volume_db=vocals_volume_db,
            )
        else:
            logger.info(f"VevoSing SVC request: mode={mode}, steps={flow_matching_steps}")
            sample_rate, audio_data = manager.vevosing_inference(
                content_wav_path=tmp_content_path,
                reference_wav_path=tmp_ref_path,
                **svc_kwargs,
            )
            output_path = _save_svc_output(audio_data, sample_rate, "vevosing")
            background_tasks.add_task(cleanup_file, output_path)
            response = FileResponse(output_path, media_type="audio/wav", filename="vevosing_output.wav")

        background_tasks.add_task(cleanup_file, tmp_content_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        return response

    except Exception as e:
        cleanup_file(tmp_content_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"VevoSing SVC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diffcomosvc")
async def diffcomosvc_inference(
    background_tasks: BackgroundTasks,
    content_audio: UploadFile = File(..., description="Source audio (content/melody)"),
    reference_audio: UploadFile = File(..., description="Reference audio (timbre)"),
    full_song_mode: bool = Form(False, description="Enable Full Song Mode (auto-separate vocals, convert, remix)"),
    vocals_volume_db: float = Form(0.0, description="Vocal volume adjustment in dB (-6 to +6)"),
):
    """
    Singing Voice Conversion using DiffComoSVC (EXPERIMENTAL).

    When full_song_mode is enabled, automatically separates vocals from instrumentals,
    converts the vocals, and remixes with the original accompaniment.
    """
    manager = ModelManager()

    content_audio = await validate_audio_file(content_audio, "content_audio", max_size=100*1024*1024)
    reference_audio = await validate_audio_file(reference_audio, "reference_audio", max_size=50*1024*1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_content:
        content = await content_audio.read()
        tmp_content.write(content)
        tmp_content_path = tmp_content.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        if full_song_mode:
            logger.info("DiffComoSVC Full Song Mode")
            response = _run_full_song_pipeline(
                manager=manager,
                content_path=tmp_content_path,
                reference_path=tmp_ref_path,
                svc_inference_fn=manager.diffcomosvc_inference,
                svc_kwargs={},
                output_prefix="diffcomosvc",
                background_tasks=background_tasks,
                vocals_volume_db=vocals_volume_db,
            )
        else:
            logger.info("DiffComoSVC request (experimental)")
            sample_rate, audio_data = manager.diffcomosvc_inference(
                content_wav_path=tmp_content_path,
                reference_wav_path=tmp_ref_path,
            )
            output_path = _save_svc_output(audio_data, sample_rate, "diffcomosvc")
            background_tasks.add_task(cleanup_file, output_path)
            response = FileResponse(output_path, media_type="audio/wav", filename="diffcomosvc_output.wav")

        background_tasks.add_task(cleanup_file, tmp_content_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        return response

    except Exception as e:
        cleanup_file(tmp_content_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"DiffComoSVC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transformersvc")
async def transformersvc_inference(
    background_tasks: BackgroundTasks,
    content_audio: UploadFile = File(..., description="Source audio (content/melody)"),
    reference_audio: UploadFile = File(..., description="Reference audio (timbre)"),
    full_song_mode: bool = Form(False, description="Enable Full Song Mode (auto-separate vocals, convert, remix)"),
    vocals_volume_db: float = Form(0.0, description="Vocal volume adjustment in dB (-6 to +6)"),
):
    """
    Singing Voice Conversion using TransformerSVC (EXPERIMENTAL).

    When full_song_mode is enabled, automatically separates vocals from instrumentals,
    converts the vocals, and remixes with the original accompaniment.
    """
    manager = ModelManager()

    content_audio = await validate_audio_file(content_audio, "content_audio", max_size=100*1024*1024)
    reference_audio = await validate_audio_file(reference_audio, "reference_audio", max_size=50*1024*1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_content:
        content = await content_audio.read()
        tmp_content.write(content)
        tmp_content_path = tmp_content.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        if full_song_mode:
            logger.info("TransformerSVC Full Song Mode")
            response = _run_full_song_pipeline(
                manager=manager,
                content_path=tmp_content_path,
                reference_path=tmp_ref_path,
                svc_inference_fn=manager.transformersvc_inference,
                svc_kwargs={},
                output_prefix="transformersvc",
                background_tasks=background_tasks,
                vocals_volume_db=vocals_volume_db,
            )
        else:
            logger.info("TransformerSVC request (experimental)")
            sample_rate, audio_data = manager.transformersvc_inference(
                content_wav_path=tmp_content_path,
                reference_wav_path=tmp_ref_path,
            )
            output_path = _save_svc_output(audio_data, sample_rate, "transformersvc")
            background_tasks.add_task(cleanup_file, output_path)
            response = FileResponse(output_path, media_type="audio/wav", filename="transformersvc_output.wav")

        background_tasks.add_task(cleanup_file, tmp_content_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        return response

    except Exception as e:
        cleanup_file(tmp_content_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"TransformerSVC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vitssvc")
async def vitssvc_inference(
    background_tasks: BackgroundTasks,
    content_audio: UploadFile = File(..., description="Source audio (content/melody)"),
    reference_audio: UploadFile = File(..., description="Reference audio (timbre)"),
    full_song_mode: bool = Form(False, description="Enable Full Song Mode (auto-separate vocals, convert, remix)"),
    vocals_volume_db: float = Form(0.0, description="Vocal volume adjustment in dB (-6 to +6)"),
):
    """
    Singing Voice Conversion using VitsSVC (EXPERIMENTAL).

    When full_song_mode is enabled, automatically separates vocals from instrumentals,
    converts the vocals, and remixes with the original accompaniment.
    """
    manager = ModelManager()

    content_audio = await validate_audio_file(content_audio, "content_audio", max_size=100*1024*1024)
    reference_audio = await validate_audio_file(reference_audio, "reference_audio", max_size=50*1024*1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_content:
        content = await content_audio.read()
        tmp_content.write(content)
        tmp_content_path = tmp_content.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        if full_song_mode:
            logger.info("VitsSVC Full Song Mode")
            response = _run_full_song_pipeline(
                manager=manager,
                content_path=tmp_content_path,
                reference_path=tmp_ref_path,
                svc_inference_fn=manager.vitssvc_inference,
                svc_kwargs={},
                output_prefix="vitssvc",
                background_tasks=background_tasks,
                vocals_volume_db=vocals_volume_db,
            )
        else:
            logger.info("VitsSVC request (experimental)")
            sample_rate, audio_data = manager.vitssvc_inference(
                content_wav_path=tmp_content_path,
                reference_wav_path=tmp_ref_path,
            )
            output_path = _save_svc_output(audio_data, sample_rate, "vitssvc")
            background_tasks.add_task(cleanup_file, output_path)
            response = FileResponse(output_path, media_type="audio/wav", filename="vitssvc_output.wav")

        background_tasks.add_task(cleanup_file, tmp_content_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        return response

    except Exception as e:
        cleanup_file(tmp_content_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"VitsSVC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multiplecontentssvc")
async def multiplecontentssvc_inference(
    background_tasks: BackgroundTasks,
    content_audio: UploadFile = File(..., description="Source audio (content/melody)"),
    reference_audio: UploadFile = File(..., description="Reference audio (timbre)"),
    full_song_mode: bool = Form(False, description="Enable Full Song Mode (auto-separate vocals, convert, remix)"),
    vocals_volume_db: float = Form(0.0, description="Vocal volume adjustment in dB (-6 to +6)"),
):
    """
    Singing Voice Conversion using MultipleContentsSVC (EXPERIMENTAL).

    When full_song_mode is enabled, automatically separates vocals from instrumentals,
    converts the vocals, and remixes with the original accompaniment.
    """
    manager = ModelManager()

    content_audio = await validate_audio_file(content_audio, "content_audio", max_size=100*1024*1024)
    reference_audio = await validate_audio_file(reference_audio, "reference_audio", max_size=50*1024*1024)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_content:
        content = await content_audio.read()
        tmp_content.write(content)
        tmp_content_path = tmp_content.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        content = await reference_audio.read()
        tmp_ref.write(content)
        tmp_ref_path = tmp_ref.name

    try:
        if full_song_mode:
            logger.info("MultipleContentsSVC Full Song Mode")
            response = _run_full_song_pipeline(
                manager=manager,
                content_path=tmp_content_path,
                reference_path=tmp_ref_path,
                svc_inference_fn=manager.multiplecontentssvc_inference,
                svc_kwargs={},
                output_prefix="multiplecontentssvc",
                background_tasks=background_tasks,
                vocals_volume_db=vocals_volume_db,
            )
        else:
            logger.info("MultipleContentsSVC request (experimental)")
            sample_rate, audio_data = manager.multiplecontentssvc_inference(
                content_wav_path=tmp_content_path,
                reference_wav_path=tmp_ref_path,
            )
            output_path = _save_svc_output(audio_data, sample_rate, "multiplecontentssvc")
            background_tasks.add_task(cleanup_file, output_path)
            response = FileResponse(output_path, media_type="audio/wav", filename="multiplecontentssvc_output.wav")

        background_tasks.add_task(cleanup_file, tmp_content_path)
        background_tasks.add_task(cleanup_file, tmp_ref_path)
        return response

    except Exception as e:
        cleanup_file(tmp_content_path)
        cleanup_file(tmp_ref_path)
        logger.error(f"MultipleContentsSVC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
