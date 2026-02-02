"""
Progress WebSocket

Real-time progress updates for inference tasks.
"""

from fastapi import WebSocket
import asyncio
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for progress updates."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, task_id: str, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if task_id not in self.active_connections:
                self.active_connections[task_id] = set()
            self.active_connections[task_id].add(websocket)
        logger.info(f"Client connected to task {task_id}")

    def disconnect(self, task_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"Client disconnected from task {task_id}")

    async def send_progress(self, task_id: str, progress: float, message: str, stage: str = ""):
        """
        Send progress update to all clients watching a task.

        Args:
            task_id: Unique task identifier
            progress: Progress percentage (0-100)
            message: Status message
            stage: Current processing stage
        """
        if task_id in self.active_connections:
            dead_connections = set()
            for connection in self.active_connections[task_id]:
                try:
                    await connection.send_json({
                        "task_id": task_id,
                        "progress": progress,
                        "message": message,
                        "stage": stage
                    })
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    dead_connections.add(connection)

            # Clean up dead connections
            for connection in dead_connections:
                self.disconnect(task_id, connection)

    async def send_complete(self, task_id: str, result_url: str = ""):
        """
        Send completion notification.

        Args:
            task_id: Unique task identifier
            result_url: URL to download result
        """
        if task_id in self.active_connections:
            for connection in list(self.active_connections[task_id]):
                try:
                    await connection.send_json({
                        "task_id": task_id,
                        "progress": 100,
                        "message": "Complete",
                        "stage": "done",
                        "result_url": result_url
                    })
                except Exception:
                    pass

    async def send_error(self, task_id: str, error: str):
        """
        Send error notification.

        Args:
            task_id: Unique task identifier
            error: Error message
        """
        if task_id in self.active_connections:
            for connection in list(self.active_connections[task_id]):
                try:
                    await connection.send_json({
                        "task_id": task_id,
                        "progress": -1,
                        "message": error,
                        "stage": "error"
                    })
                except Exception:
                    pass


# Global connection manager instance
manager = ConnectionManager()
