"""
File Upload Validation

Validation utilities for audio file uploads.
Ensures file size limits, valid formats, and basic security checks.
"""

import os
import logging
from typing import Optional
from fastapi import UploadFile, HTTPException, status

# Optional magic import for file type detection
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logging.getLogger(__name__).warning("python-magic not installed. File type detection disabled.")

logger = logging.getLogger(__name__)

# Upload limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB max upload
MAX_AUDIO_DURATION = 300  # 5 minutes max (in seconds)
ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",  # MP3
    "audio/mp3",
    "audio/flac",
    "audio/ogg",
    "audio/x-m4a",
    "audio/mp4",
}
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class UploadValidator:
    """Validator for audio file uploads."""

    def __init__(
        self,
        max_size: int = MAX_FILE_SIZE,
        allowed_types: set = None,
        allowed_extensions: set = None,
    ):
        self.max_size = max_size
        self.allowed_types = allowed_types or ALLOWED_AUDIO_TYPES
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS

    async def validate(self, file: UploadFile, field_name: str = "file") -> UploadFile:
        """
        Validate an uploaded file.

        Args:
            file: The uploaded file
            field_name: Field name for error messages

        Returns:
            The validated file

        Raises:
            HTTPException: If validation fails
        """
        # Check file exists
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name}: No file provided",
            )

        # Check file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in self.allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name}: Invalid file extension '{ext}'. Allowed: {', '.join(self.allowed_extensions)}",
            )

        # Check content type hint
        if file.content_type and not any(
            allowed in file.content_type for allowed in self.allowed_types
        ):
            logger.warning(
                f"Content type mismatch: {file.content_type} for {file.filename}"
            )

        # Read and validate file size
        content = await file.read()
        file_size = len(content)

        if file_size > self.max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{field_name}: File too large ({file_size / 1024 / 1024:.1f}MB). Max: {self.max_size / 1024 / 1024}MB",
            )

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{field_name}: Empty file",
            )

        # Validate magic bytes (actual file content) if available
        if HAS_MAGIC:
            try:
                detected = magic.from_buffer(content, mime=True)
                if detected not in self.allowed_types:
                    # Some systems report slightly different MIME types
                    if not any(allowed in detected for allowed in ["audio", "wav", "mpeg"]):
                        logger.warning(
                            f"Magic bytes mismatch: {detected} for {file.filename}"
                        )
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"{field_name}: File content doesn't match extension. Detected: {detected}",
                        )
            except Exception as e:
                logger.error(f"Magic validation error: {e}")
                # Continue anyway, magic might not work properly
                detected = "unknown"
        else:
            detected = "unknown"

        # Reset file pointer for reading
        await file.seek(0)

        logger.info(
            f"Validated upload: {file.filename} ({file_size / 1024:.1f}KB, type: {detected if 'detected' in dir() else 'unknown'})"
        )

        return file


# Default validator instance
default_validator = UploadValidator()


async def validate_audio_file(
    file: UploadFile,
    field_name: str = "file",
    max_size: Optional[int] = None,
) -> UploadFile:
    """
    Convenience function to validate an audio file.

    Args:
        file: The uploaded file
        field_name: Field name for error messages
        max_size: Optional custom max size

    Returns:
        The validated file
    """
    validator = default_validator
    if max_size:
        validator = UploadValidator(max_size=max_size)

    return await validator.validate(file, field_name)
