"""
Health Check Routes

Provides health check and model status endpoints.
"""

from fastapi import APIRouter
from typing import Dict, Any
import torch
import logging

from ..models.manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint.

    Returns:
        dict: Health status
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@router.get("/models/status")
async def models_status() -> Dict[str, Any]:
    """
    Get status of all models.

    Returns:
        dict: Model loading status and available CUDA devices
    """
    manager = ModelManager()

    return {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "N/A",
        "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A",
        "cuda_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A",
        "models": {
            "maskgct": {
                "loaded": manager._maskgct_loaded,
                "name": "MaskGCT",
                "description": "Zero-shot TTS with neural codec language model"
            },
            "dualcodec_valle": {
                "loaded": manager._dualcodec_valle_loaded,
                "name": "DualCodec-VALLE",
                "description": "Fast 12.5Hz codec TTS"
            },
            "vevo_tts": {
                "loaded": manager._vevo_tts_loaded,
                "name": "Vevo TTS",
                "description": "Style/timbre controllable TTS"
            },
            "vevo_vc": {
                "loaded": manager._vevo_vc_loaded,
                "name": "Vevo VC",
                "description": "Voice conversion with style control"
            },
        }
    }


@router.post("/models/unload/{model_name}")
async def unload_model(model_name: str) -> Dict[str, str]:
    """
    Unload a specific model to free GPU memory.

    Args:
        model_name: Name of model to unload (maskgct, dualcodec_valle, vevo_tts, vevo_vc)

    Returns:
        dict: Unload status
    """
    manager = ModelManager()

    try:
        manager.unload_model(model_name)
        return {"status": "success", "message": f"Model {model_name} unloaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
