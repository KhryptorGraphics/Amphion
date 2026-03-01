"""
Presets Routes

CRUD endpoints for managing model configuration presets.
Presets are stored as individual JSON files in the output/web/presets/ directory.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import json
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# File-based storage directory (relative to project root)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
PRESETS_DIR = os.path.join(_PROJECT_ROOT, "output", "web", "presets")
os.makedirs(PRESETS_DIR, exist_ok=True)


# ===========================
# Pydantic Models
# ===========================


class PresetCreate(BaseModel):
    """Request model for creating a preset."""

    name: str = Field(..., description="Preset name")
    model_id: str = Field(..., description="Model identifier (e.g., maskgct, vevo_tts)")
    description: Optional[str] = Field(None, description="Preset description")
    parameters: Dict[str, Any] = Field(
        ..., description="Model configuration parameters"
    )


class PresetUpdate(BaseModel):
    """Request model for updating a preset."""

    name: Optional[str] = Field(None, description="Preset name")
    description: Optional[str] = Field(None, description="Preset description")
    parameters: Optional[Dict[str, Any]] = Field(
        None, description="Model configuration parameters"
    )


class Preset(BaseModel):
    """Preset response model."""

    id: str
    name: str
    model_id: str
    description: Optional[str]
    parameters: Dict[str, Any]
    created_at: str
    updated_at: str


# ===========================
# Storage Utility Functions
# ===========================


def list_presets(model_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all presets, optionally filtered by model_id.

    Args:
        model_id: Optional model identifier to filter by

    Returns:
        List of preset dictionaries sorted by created_at descending
    """
    presets = []
    try:
        for filename in os.listdir(PRESETS_DIR):
            if filename.endswith(".json"):
                file_path = os.path.join(PRESETS_DIR, filename)
                try:
                    with open(file_path, "r") as f:
                        preset = json.load(f)
                    presets.append(preset)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to read preset file {filename}: {e}")
    except OSError as e:
        logger.error(f"Failed to list presets directory: {e}")

    if model_id:
        presets = [p for p in presets if p.get("model_id") == model_id]

    presets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return presets


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single preset by ID.

    Args:
        preset_id: Preset identifier

    Returns:
        Preset dictionary or None if not found
    """
    file_path = os.path.join(PRESETS_DIR, f"{preset_id}.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read preset {preset_id}: {e}")
        return None


def save_preset(preset: Dict[str, Any]) -> bool:
    """
    Save a preset to disk as a JSON file.

    Args:
        preset: Preset dictionary to save (must include 'id' key)

    Returns:
        True if successful, False otherwise
    """
    file_path = os.path.join(PRESETS_DIR, f"{preset['id']}.json")
    try:
        with open(file_path, "w") as f:
            json.dump(preset, f, indent=2)
        return True
    except OSError as e:
        logger.error(f"Failed to save preset {preset['id']}: {e}")
        return False


def delete_preset(preset_id: str) -> bool:
    """
    Delete a preset file from disk.

    Args:
        preset_id: Preset identifier

    Returns:
        True if deleted, False if not found or error
    """
    file_path = os.path.join(PRESETS_DIR, f"{preset_id}.json")
    if not os.path.exists(file_path):
        return False
    try:
        os.remove(file_path)
        return True
    except OSError as e:
        logger.error(f"Failed to delete preset {preset_id}: {e}")
        return False


# ===========================
# Preset Endpoints
# ===========================


@router.get("/presets", response_model=List[Preset])
async def get_presets(
    model_id: Optional[str] = Query(None, description="Filter by model ID")
) -> List[Preset]:
    """
    List all saved presets, optionally filtered by model_id.

    Args:
        model_id: Optional query parameter to filter presets by model

    Returns:
        List of presets sorted by creation date descending
    """
    presets = list_presets(model_id=model_id)
    return [Preset(**p) for p in presets]


@router.get("/presets/{preset_id}", response_model=Preset)
async def get_preset_by_id(preset_id: str) -> Preset:
    """
    Get a specific preset by ID.

    Args:
        preset_id: Preset identifier

    Returns:
        Preset details
    """
    preset = get_preset(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")
    return Preset(**preset)


@router.post("/presets", response_model=Preset, status_code=201)
async def create_preset(preset_create: PresetCreate) -> Preset:
    """
    Create a new configuration preset.

    Args:
        preset_create: Preset creation data including name, model_id, and parameters

    Returns:
        Created preset with generated ID and timestamps
    """
    preset_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    preset = {
        "id": preset_id,
        "name": preset_create.name,
        "model_id": preset_create.model_id,
        "description": preset_create.description,
        "parameters": preset_create.parameters,
        "created_at": now,
        "updated_at": now,
    }

    if not save_preset(preset):
        raise HTTPException(status_code=500, detail="Failed to save preset")

    logger.info(
        f"Created preset {preset_id}: {preset_create.name} for model {preset_create.model_id}"
    )
    return Preset(**preset)


@router.put("/presets/{preset_id}", response_model=Preset)
async def update_preset(preset_id: str, preset_update: PresetUpdate) -> Preset:
    """
    Update an existing preset.

    Args:
        preset_id: Preset identifier
        preset_update: Fields to update (all optional)

    Returns:
        Updated preset
    """
    preset = get_preset(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")

    if preset_update.name is not None:
        preset["name"] = preset_update.name
    if preset_update.description is not None:
        preset["description"] = preset_update.description
    if preset_update.parameters is not None:
        preset["parameters"] = preset_update.parameters

    preset["updated_at"] = datetime.utcnow().isoformat()

    if not save_preset(preset):
        raise HTTPException(status_code=500, detail="Failed to save preset")

    logger.info(f"Updated preset {preset_id}")
    return Preset(**preset)


@router.delete("/presets/{preset_id}", status_code=204)
async def remove_preset(preset_id: str) -> None:
    """
    Delete a preset.

    Args:
        preset_id: Preset identifier
    """
    preset = get_preset(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found")

    if not delete_preset(preset_id):
        raise HTTPException(status_code=500, detail="Failed to delete preset")

    logger.info(f"Deleted preset {preset_id}")
    return None
