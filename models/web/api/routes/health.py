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


@router.get("/gpu/status")
async def gpu_status() -> Dict[str, Any]:
    """
    Get GPU status and CUDA information.

    Returns:
        dict: GPU status, CUDA availability, and memory usage
    """
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

            return {
                "status": "available",
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": device_name,
                "memory": {
                    "allocated_gb": round(memory_allocated, 2),
                    "reserved_gb": round(memory_reserved, 2),
                    "total_gb": round(total_memory, 2),
                    "free_gb": round(total_memory - memory_allocated, 2)
                }
            }
        else:
            return {
                "status": "unavailable",
                "cuda_available": False,
                "device_count": 0,
                "message": "CUDA not available - running on CPU"
            }
    except Exception as e:
        return {
            "status": "error",
            "cuda_available": False,
            "error": str(e)
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
            "noro": {
                "loaded": manager._noro_loaded,
                "name": "Noro",
                "description": "Noise-robust voice conversion"
            },
            "metis": {
                "loaded": manager._metis_loaded,
                "name": "Metis",
                "description": "Unified foundation model for TTS, VC, SE, TSE"
            },
            "vevosing": {
                "loaded": manager._vevosing_loaded,
                "name": "VevoSing",
                "description": "Singing voice conversion with timbre/style control"
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
