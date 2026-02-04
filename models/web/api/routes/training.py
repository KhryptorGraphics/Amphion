# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Training API Routes

API endpoints for training job management, monitoring, and checkpoint handling.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import json
import uuid

import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["training"])

# In-memory job storage (replace with database in production)
training_jobs: Dict[str, Dict[str, Any]] = {}


# ===========================
# Data Models
# ===========================

class TrainingJobCreate(BaseModel):
    """Request model for creating a training job."""
    model_type: str = Field(..., description="Type of model to train (tts, svc, vc, codec, vocoder)")
    model_name: str = Field(..., description="Specific model architecture (maskgct, vits, etc.)")
    dataset_path: str = Field(..., description="Path to training dataset")
    config: Optional[Dict[str, Any]] = Field(None, description="Training configuration overrides")
    exp_name: str = Field(..., description="Experiment name for this training run")
    description: Optional[str] = Field(None, description="Optional job description")
    priority: int = Field(5, ge=1, le=10, description="Job priority (1-10, lower is higher priority)")


class TrainingJobResponse(BaseModel):
    """Response model for training job."""
    id: str
    status: str  # pending, running, completed, failed, cancelled
    model_type: str
    model_name: str
    exp_name: str
    description: Optional[str]
    priority: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: Optional[float]  # 0-100
    current_step: Optional[int]
    total_steps: Optional[int]
    current_loss: Optional[float]
    best_loss: Optional[float]
    checkpoint_count: int
    error_message: Optional[str]


class TrainingJobUpdate(BaseModel):
    """Request model for updating a training job."""
    status: Optional[str] = None
    priority: Optional[int] = None
    description: Optional[str] = None


class CheckpointInfo(BaseModel):
    """Information about a checkpoint."""
    path: str
    step: int
    loss: Optional[float]
    created_at: str
    size_bytes: int


# ===========================
# Helper Functions
# ===========================

def _get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status from storage."""
    return training_jobs.get(job_id)


def _update_job_status(job_id: str, updates: Dict[str, Any]):
    """Update job status in storage."""
    if job_id in training_jobs:
        training_jobs[job_id].update(updates)
        training_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()


def _get_checkpoints(exp_name: str) -> List[CheckpointInfo]:
    """Get list of checkpoints for an experiment."""
    checkpoints = []
    ckpt_dir = os.path.join("ckpts", exp_name)

    if not os.path.exists(ckpt_dir):
        return checkpoints

    for filename in os.listdir(ckpt_dir):
        if filename.endswith(('.pt', '.ckpt', '.pth', '.safetensors')):
            path = os.path.join(ckpt_dir, filename)
            stat = os.stat(path)

            # Try to extract step number from filename
            step = 0
            try:
                # Common patterns: model_step_1000.pt, checkpoint-1000.pt, etc.
                import re
                match = re.search(r'[\-_](\d+)', filename)
                if match:
                    step = int(match.group(1))
            except:
                pass

            checkpoints.append(CheckpointInfo(
                path=path,
                step=step,
                loss=None,  # Would need to load checkpoint to get loss
                created_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                size_bytes=stat.st_size
            ))

    # Sort by step number
    checkpoints.sort(key=lambda x: x.step)
    return checkpoints


# ===========================
# Training Job Endpoints
# ===========================

@router.post("/jobs", response_model=TrainingJobResponse, status_code=201)
async def create_training_job(job: TrainingJobCreate):
    """
    Create a new training job.

    The job will be queued and started when resources are available.
    """
    job_id = str(uuid.uuid4())

    now = datetime.utcnow().isoformat()
    training_jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "model_type": job.model_type,
        "model_name": job.model_name,
        "exp_name": job.exp_name,
        "description": job.description,
        "priority": job.priority,
        "dataset_path": job.dataset_path,
        "config": job.config or {},
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "current_step": 0,
        "total_steps": None,
        "current_loss": None,
        "best_loss": None,
        "checkpoint_count": 0,
        "error_message": None,
        "process_id": None,
    }

    logger.info(f"Created training job {job_id} for {job.model_type}/{job.model_name}")

    return TrainingJobResponse(**training_jobs[job_id])


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List training jobs with optional filtering."""
    jobs = list(training_jobs.values())

    # Apply filters
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    if model_type:
        jobs = [j for j in jobs if j["model_type"] == model_type]

    # Sort by priority then created_at
    jobs.sort(key=lambda x: (x["priority"], x["created_at"]))

    # Apply pagination
    jobs = jobs[offset:offset + limit]

    return [TrainingJobResponse(**j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str):
    """Get details of a specific training job."""
    job = _get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    return TrainingJobResponse(**job)


@router.patch("/jobs/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(job_id: str, update: TrainingJobUpdate):
    """Update a training job (priority, description, etc.)."""
    job = _get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    updates = {}
    if update.priority is not None:
        updates["priority"] = update.priority
    if update.description is not None:
        updates["description"] = update.description

    _update_job_status(job_id, updates)

    return TrainingJobResponse(**training_jobs[job_id])


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_training_job(job_id: str):
    """Cancel a training job."""
    job = _get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    if job["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail=f"Job {job_id} is already {job['status']}")

    # Update status to cancelled
    _update_job_status(job_id, {
        "status": "cancelled",
        "completed_at": datetime.utcnow().isoformat()
    })

    logger.info(f"Cancelled training job {job_id}")

    return None


# ===========================
# Training Monitor Endpoints
# ===========================

@router.get("/jobs/{job_id}/monitor")
async def get_training_monitor(job_id: str):
    """
    Get training progress and metrics for a job.

    Returns loss curves, step progress, and other training metrics.
    """
    job = _get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Get loss history from log file if available
    loss_history = []
    log_file = os.path.join("ckpts", job["exp_name"], "training.log")

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        if 'step' in log_entry and 'loss' in log_entry:
                            loss_history.append({
                                'step': log_entry['step'],
                                'loss': log_entry['loss'],
                                'timestamp': log_entry.get('timestamp')
                            })
                    except:
                        pass
        except:
            pass

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "current_step": job["current_step"],
        "total_steps": job["total_steps"],
        "current_loss": job["current_loss"],
        "best_loss": job["best_loss"],
        "loss_history": loss_history[-100:] if len(loss_history) > 100 else loss_history,
        "eta_seconds": None,  # Would calculate based on progress rate
    }


# ===========================
# Checkpoint Endpoints
# ===========================

@router.get("/jobs/{job_id}/checkpoints", response_model=List[CheckpointInfo])
async def list_checkpoints(job_id: str):
    """List all checkpoints for a training job."""
    job = _get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    checkpoints = _get_checkpoints(job["exp_name"])

    return checkpoints


@router.get("/jobs/{job_id}/checkpoints/latest")
async def get_latest_checkpoint(job_id: str):
    """Get information about the latest checkpoint."""
    job = _get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    checkpoints = _get_checkpoints(job["exp_name"])
    if not checkpoints:
        raise HTTPException(status_code=404, detail="No checkpoints found")

    return checkpoints[-1]  # Latest checkpoint


@router.get("/jobs/{job_id}/checkpoints/{checkpoint_id}/download")
async def download_checkpoint(job_id: str, checkpoint_id: str):
    """Download a specific checkpoint."""
    job = _get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job {job_id} not found")

    # Find checkpoint
    checkpoints = _get_checkpoints(job["exp_name"])
    checkpoint = None
    for ckpt in checkpoints:
        if str(ckpt.step) == checkpoint_id or os.path.basename(ckpt.path) == checkpoint_id:
            checkpoint = ckpt
            break

    if not checkpoint:
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found")

    if not os.path.exists(checkpoint.path):
        raise HTTPException(status_code=404, detail="Checkpoint file not found on disk")

    return FileResponse(
        checkpoint.path,
        filename=os.path.basename(checkpoint.path),
        media_type='application/octet-stream'
    )


# ===========================
# WebSocket for Real-time Updates
# ===========================

from fastapi import WebSocket

@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time training updates.

    Connect to receive live progress updates during training.
    """
    await websocket.accept()

    job = _get_job_status(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    try:
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "status": job["status"]
        })

        # Keep connection open and send updates periodically
        import asyncio
        while True:
            job = _get_job_status(job_id)
            if not job:
                break

            await websocket.send_json({
                "type": "update",
                "status": job["status"],
                "progress": job["progress"],
                "current_step": job["current_step"],
                "current_loss": job["current_loss"],
                "timestamp": datetime.utcnow().isoformat()
            })

            # Check if job is complete
            if job["status"] in ["completed", "failed", "cancelled"]:
                await websocket.send_json({
                    "type": "complete",
                    "status": job["status"],
                    "final_loss": job["current_loss"]
                })
                break

            await asyncio.sleep(5)  # Update every 5 seconds

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await websocket.close()
