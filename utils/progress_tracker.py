"""
Progress Tracker

A reusable progress tracking module with ETA calculation for batch audio processing tasks.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress of batch processing tasks with ETA calculation.

    This class provides a simple way to track progress of batch operations,
    calculating estimated time of arrival (ETA) based on elapsed time and
    completion rate.

    Attributes:
        total: Total number of items to process
        current: Current number of items processed
        start_time: Timestamp when tracking started
    """

    def __init__(self, total: int):
        """Initialize the progress tracker.

        Args:
            total: Total number of items to process
        """
        if total <= 0:
            raise ValueError("Total must be a positive integer")

        self.total = total
        self.current = 0
        self.start_time: Optional[float] = None
        self._last_update_time: Optional[float] = None

    def start(self):
        """Start the progress tracking."""
        self.start_time = time.time()
        self._last_update_time = self.start_time
        logger.debug(f"Progress tracking started for {self.total} items")

    def update(self, amount: int = 1):
        """Update the progress by a given amount.

        Args:
            amount: Number of items processed since last update (default: 1)
        """
        if self.start_time is None:
            self.start()

        self.current += amount
        self._last_update_time = time.time()

        if self.current > self.total:
            self.current = self.total

        logger.debug(f"Progress updated: {self.current}/{self.total}")

    def set_current(self, value: int):
        """Set the current progress to a specific value.

        Args:
            value: The current number of items processed
        """
        if self.start_time is None:
            self.start()

        self.current = max(0, min(value, self.total))
        self._last_update_time = time.time()

    def get_progress(self) -> float:
        """Get the current progress percentage.

        Returns:
            Progress percentage (0-100)
        """
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100.0

    def get_eta(self) -> Optional[float]:
        """Get the estimated time remaining in seconds.

        Returns:
            Estimated seconds remaining, or None if not enough data
        """
        if self.start_time is None or self.current == 0:
            return None

        elapsed = time.time() - self.start_time

        if self.current >= self.total:
            return 0.0

        # Calculate rate: items per second
        rate = self.current / elapsed

        if rate <= 0:
            return None

        # Calculate remaining time
        remaining_items = self.total - self.current
        eta = remaining_items / rate

        return eta

    def get_elapsed(self) -> float:
        """Get the elapsed time in seconds.

        Returns:
            Elapsed seconds since start, or 0 if not started
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_eta_formatted(self) -> str:
        """Get the ETA as a human-readable string.

        Returns:
            Formatted ETA string (e.g., "5m 30s" or "Unknown")
        """
        eta = self.get_eta()

        if eta is None:
            return "Unknown"

        if eta < 60:
            return f"{int(eta)}s"
        elif eta < 3600:
            minutes = int(eta // 60)
            seconds = int(eta % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            return f"{hours}h {minutes}m"

    def is_complete(self) -> bool:
        """Check if the task is complete.

        Returns:
            True if all items have been processed
        """
        return self.current >= self.total

    def reset(self):
        """Reset the tracker to initial state."""
        self.current = 0
        self.start_time = None
        self._last_update_time = None
        logger.debug("Progress tracker reset")

    def get_status(self) -> dict:
        """Get the current status as a dictionary.

        Returns:
            Dictionary containing progress information
        """
        return {
            "current": self.current,
            "total": self.total,
            "progress": self.get_progress(),
            "eta_seconds": self.get_eta(),
            "eta_formatted": self.get_eta_formatted(),
            "elapsed_seconds": self.get_elapsed(),
            "is_complete": self.is_complete(),
        }
