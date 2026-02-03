"""
Amphion FastAPI Server

Main FastAPI application entry point for Amphion TTS/VC models.
Provides REST API endpoints and WebSocket support for real-time progress updates.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging

from .routes import tts, vc, svc, health, evaluation
from .websocket.progress import manager
from .auth import AuthMiddleware
from .rate_limit import RateLimitMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Amphion API",
    version="1.0.0",
    description="REST API for Amphion TTS and Voice Conversion models",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aphion.giggahost.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware - Rate limiting and Auth
# Order matters: Rate limit first, then auth
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Ensure output directory exists
os.makedirs("/home/kp/repo2/Amphion/output/web", exist_ok=True)

# Include routers
app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])
app.include_router(vc.router, prefix="/api/vc", tags=["Voice Conversion"])
app.include_router(svc.router, prefix="/api/svc", tags=["Singing Voice Conversion"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(health.router, prefix="/api", tags=["Health"])


# Mount React frontend static files
react_build_dir = os.path.join(os.path.dirname(__file__), "../react/dist")
if os.path.exists(react_build_dir):
    app.mount("/", StaticFiles(directory=react_build_dir, html=True), name="frontend")
    logger.info(f"Serving React frontend from: {react_build_dir}")
else:
    logger.warning(f"React build directory not found: {react_build_dir}")


@app.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time progress updates.

    Clients can connect to /ws/progress/{task_id} to receive updates
    about a specific inference task.
    """
    await manager.connect(task_id, websocket)
    try:
        while True:
            # Keep connection alive, receive any messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(task_id, websocket)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Amphion API server starting...")
    logger.info("Output directory: /home/kp/repo2/Amphion/output/web")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Amphion API server shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "models.web.api.main:app",
        host="127.0.0.1",
        port=14555,
        reload=False,
        log_level="info"
    )
