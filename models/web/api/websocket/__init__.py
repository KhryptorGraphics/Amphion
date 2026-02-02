"""
WebSocket Support

Real-time progress updates for long-running inference tasks.
"""

from .progress import manager, ConnectionManager

__all__ = ["manager", "ConnectionManager"]
