"""
Codec Routes

Endpoints for neural audio codec encode/decode operations.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import logging
import json

from ..models.manager import ModelManager
from ..upload_validation import validate_audio_file

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


@router.post("/dualcodec/encode")
async def dualcodec_encode(
    audio: UploadFile = File(..., description="Audio file to encode"),
):
    """
    Encode audio to discrete tokens using DualCodec.

    DualCodec compresses audio into discrete tokens at 12Hz with 12,000 tokens/second.
    The encoded tokens can be stored or transmitted efficiently and decoded back to audio.

    Args:
        audio: Audio file to encode (WAV, MP3, etc.)

    Returns:
        JSONResponse: Dictionary containing:
            - tokens: List of discrete token codes
            - codec_type: "dualcodec"
            - sample_rate: Original sample rate
    """
    manager = ModelManager()

    # Validate uploaded audio file
    audio = await validate_audio_file(audio, "audio")

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        content = await audio.read()
        tmp_audio.write(content)
        tmp_audio_path = tmp_audio.name

    try:
        logger.info(f"DualCodec encode request")

        # Encode audio
        result = manager.codec_encode(
            audio_path=tmp_audio_path,
            codec_type="dualcodec",
        )

        # Cleanup temp file
        cleanup_file(tmp_audio_path)

        return JSONResponse(content=result)

    except Exception as e:
        cleanup_file(tmp_audio_path)
        logger.error(f"DualCodec encode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dualcodec/decode")
async def dualcodec_decode(
    background_tasks: BackgroundTasks,
    tokens: str = Form(..., description="JSON-encoded token array"),
):
    """
    Decode discrete tokens to audio using DualCodec.

    Args:
        tokens: JSON-encoded token array (e.g., "[[1, 2, 3], [4, 5, 6]]")

    Returns:
        FileResponse: Decoded audio file (WAV format, 16kHz)
    """
    manager = ModelManager()

    try:
        # Parse tokens from JSON
        tokens_data = json.loads(tokens)

        logger.info(f"DualCodec decode request: tokens shape check")

        # Decode tokens
        output_path = manager.codec_decode(
            tokens=tokens_data,
            codec_type="dualcodec",
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="dualcodec_decoded.wav",
            background=background_tasks
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid token JSON: {str(e)}")
    except Exception as e:
        logger.error(f"DualCodec decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facodec/encode")
async def facodec_encode(
    audio: UploadFile = File(..., description="Audio file to encode"),
):
    """
    Encode audio to discrete tokens using FACodec (FAst Codec).

    FACodec is a fast neural audio codec optimized for real-time applications.

    Args:
        audio: Audio file to encode (WAV, MP3, etc.)

    Returns:
        JSONResponse: Dictionary containing:
            - tokens: List of discrete token codes
            - codec_type: "facodec"
            - sample_rate: Original sample rate
    """
    manager = ModelManager()

    # Validate uploaded audio file
    audio = await validate_audio_file(audio, "audio")

    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        content = await audio.read()
        tmp_audio.write(content)
        tmp_audio_path = tmp_audio.name

    try:
        logger.info(f"FACodec encode request")

        # Encode audio
        result = manager.codec_encode(
            audio_path=tmp_audio_path,
            codec_type="facodec",
        )

        # Cleanup temp file
        cleanup_file(tmp_audio_path)

        return JSONResponse(content=result)

    except Exception as e:
        cleanup_file(tmp_audio_path)
        logger.error(f"FACodec encode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/facodec/decode")
async def facodec_decode(
    background_tasks: BackgroundTasks,
    tokens: str = Form(..., description="JSON-encoded token array"),
):
    """
    Decode discrete tokens to audio using FACodec.

    Args:
        tokens: JSON-encoded token array (e.g., "[[1, 2, 3], [4, 5, 6]]")

    Returns:
        FileResponse: Decoded audio file (WAV format, 16kHz)
    """
    manager = ModelManager()

    try:
        # Parse tokens from JSON
        tokens_data = json.loads(tokens)

        logger.info(f"FACodec decode request")

        # Decode tokens
        output_path = manager.codec_decode(
            tokens=tokens_data,
            codec_type="facodec",
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_file, output_path)

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="facodec_decoded.wav",
            background=background_tasks
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid token JSON: {str(e)}")
    except Exception as e:
        logger.error(f"FACodec decode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
