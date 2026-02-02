# Tech Stack

## Backend

- **Language:** Python 3.10+
- **Framework:** FastAPI with uvicorn
- **WebSocket:** FastAPI WebSocket for real-time progress
- **Models:** PyTorch with CUDA 13.0 (Jetson Thor)
- **Port:** 17862

## Frontend

- **Framework:** Next.js 15 with App Router
- **Language:** TypeScript
- **Styling:** Tailwind CSS
- **UI Components:** Radix UI primitives
- **Icons:** Lucide React
- **HTTP Client:** Axios
- **Port:** 3001

## Infrastructure

- **Web Server:** Apache 2.4 with mod_proxy
- **SSL:** Let's Encrypt
- **Process Manager:** systemd
- **GPU:** NVIDIA Thor (sm_110)

## Services

| Service | Port | Path | Description |
|---------|------|------|-------------|
| amphion-api | 17862 | /api/ | FastAPI backend |
| amphion-frontend | 3001 | /react/ | Next.js frontend |
| amphion-gradio | 17860 | /gradio/ | Legacy Gradio demo |
