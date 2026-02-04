# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset API Routes

API endpoints for dataset management and preprocessing.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import json
import uuid
import shutil

from ..utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])

# In-memory dataset storage
datasets: Dict[str, Dict[str, Any]] = {}
DATASETS_DIR = "/home/kp/repo2/Amphion/data/datasets"

os.makedirs(DATASETS_DIR, exist_ok=True)


# ===========================
# Data Models
# ===========================

class DatasetCreate(BaseModel):
    """Request model for creating a dataset."""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    dataset_type: str = Field(..., description="Type: tts, svc, vc, codec")
    sample_rate: int = Field(16000, description="Audio sample rate")
    language: Optional[str] = Field("en", description="Primary language")


class DatasetResponse(BaseModel):
    """Response model for dataset."""
    id: str
    name: str
    description: Optional[str]
    dataset_type: str
    sample_rate: int
    language: str
    status: str  # uploading, processing, ready, error
    file_count: int
    total_duration_seconds: Optional[float]
    created_at: str
    updated_at: str
    path: str


class DatasetSample(BaseModel):
    """Information about a dataset sample."""
    id: str
    filename: str
    duration_seconds: Optional[float]
    sample_rate: int
    text: Optional[str]  # For TTS
    speaker_id: Optional[str]
    language: Optional[str]


class PreprocessRequest(BaseModel):
    """Request to start preprocessing."""
    preprocessor: str = Field(..., description="Preprocessor to use")
    config: Optional[Dict[str, Any]] = Field(None, description="Preprocessing config")


# ===========================
# Dataset Endpoints
# ===========================

@router.post("", response_model=DatasetResponse, status_code=201)
async def create_dataset(dataset: DatasetCreate):
    """Create a new dataset."""
    dataset_id = str(uuid.uuid4())
    dataset_path = os.path.join(DATASETS_DIR, dataset_id)
    os.makedirs(dataset_path, exist_ok=True)

    now = datetime.utcnow().isoformat()
    datasets[dataset_id] = {
        "id": dataset_id,
        "name": dataset.name,
        "description": dataset.description,
        "dataset_type": dataset.dataset_type,
        "sample_rate": dataset.sample_rate,
        "language": dataset.language,
        "status": "uploading",
        "file_count": 0,
        "total_duration_seconds": None,
        "created_at": now,
        "updated_at": now,
        "path": dataset_path,
    }

    logger.info(f"Created dataset {dataset_id}: {dataset.name}")
    return DatasetResponse(**datasets[dataset_id])


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(
    dataset_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """List all datasets."""
    result = list(datasets.values())

    if dataset_type:
        result = [d for d in result if d["dataset_type"] == dataset_type]
    if status:
        result = [d for d in result if d["status"] == status]

    result.sort(key=lambda x: x["created_at"], reverse=True)

    return [DatasetResponse(**d) for d in result[offset:offset + limit]]


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """Get dataset details."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse(**datasets[dataset_id])


@router.delete("/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str):
    """Delete a dataset and its files."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = datasets[dataset_id]["path"]

    # Remove files
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    del datasets[dataset_id]
    logger.info(f"Deleted dataset {dataset_id}")

    return None


@router.post("/{dataset_id}/upload")
async def upload_audio(
    dataset_id: str,
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    speaker_id: Optional[str] = Form(None)
):
    """Upload an audio file to a dataset."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = datasets[dataset_id]
    dataset_path = dataset["path"]

    # Save file
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(dataset_path, filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Update metadata
    dataset["file_count"] += 1
    dataset["updated_at"] = datetime.utcnow().isoformat()

    # Save metadata sidecar
    metadata = {
        "filename": filename,
        "text": text,
        "speaker_id": speaker_id,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    metadata_path = os.path.join(dataset_path, f"{file_id}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    logger.info(f"Uploaded {filename} to dataset {dataset_id}")

    return {"file_id": file_id, "dataset_id": dataset_id, "filename": filename}


@router.get("/{dataset_id}/samples", response_model=List[DatasetSample])
async def list_samples(dataset_id: str, limit: int = 50, offset: int = 0):
    """List samples in a dataset."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = datasets[dataset_id]["path"]
    samples = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(('.wav', '.mp3', '.flac')):
            file_id = filename.split('_')[0]
            metadata_path = os.path.join(dataset_path, f"{file_id}.json")
            metadata = {}

            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)

            # Get audio info
            import soundfile as sf
            file_path = os.path.join(dataset_path, filename)
            try:
                info = sf.info(file_path)
                duration = info.duration
                sr = info.samplerate
            except:
                duration = None
                sr = datasets[dataset_id]["sample_rate"]

            samples.append(DatasetSample(
                id=file_id,
                filename=filename,
                duration_seconds=duration,
                sample_rate=sr,
                text=metadata.get("text"),
                speaker_id=metadata.get("speaker_id"),
                language=datasets[dataset_id]["language"]
            ))

    return samples[offset:offset + limit]


@router.get("/{dataset_id}/samples/{sample_id}/preview")
async def preview_sample(dataset_id: str, sample_id: str):
    """Get audio file for preview."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset_path = datasets[dataset_id]["path"]

    # Find file
    for filename in os.listdir(dataset_path):
        if filename.startswith(sample_id) and filename.endswith(('.wav', '.mp3', '.flac')):
            file_path = os.path.join(dataset_path, filename)
            return FileResponse(file_path, media_type="audio/wav")

    raise HTTPException(status_code=404, detail="Sample not found")


@router.post("/{dataset_id}/preprocess")
async def preprocess_dataset(
    dataset_id: str,
    request: PreprocessRequest,
    background_tasks: BackgroundTasks
):
    """Start dataset preprocessing."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    datasets[dataset_id]["status"] = "processing"
    datasets[dataset_id]["updated_at"] = datetime.utcnow().isoformat()

    # In real implementation, this would run preprocessing in background
    logger.info(f"Started preprocessing dataset {dataset_id} with {request.preprocessor}")

    return {
        "dataset_id": dataset_id,
        "status": "processing",
        "preprocessor": request.preprocessor,
        "message": "Preprocessing started (async implementation needed)"
    }


@router.get("/{dataset_id}/preprocess/status")
async def get_preprocess_status(dataset_id: str):
    """Get preprocessing status."""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "dataset_id": dataset_id,
        "status": datasets[dataset_id]["status"],
        "progress": None,  # Would track actual progress
    }
