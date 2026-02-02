# Amphion Web Frontend Upgrade

## Project Overview

**Name:** Amphion Web Frontend
**Description:** Custom React frontend replacing Gradio iframe embedding for the Amphion audio generation toolkit

## Problem Statement

The current web interface uses Gradio iframes which limit customization, expose only 4 models, and lack features like audio history, comparison, and batch processing.

## Target Users

- Researchers using Amphion for TTS/VC experiments
- Developers integrating audio generation into applications
- End users wanting to generate speech or convert voices

## Key Goals

1. Expose ALL Amphion capabilities (TTS, VC, SVC, TTA, Codecs, Vocoders, Metrics)
2. Provide beautiful, modern UI with audio visualization
3. Real-time progress updates via WebSocket
4. Audio history, comparison, and batch processing features

## Current State

- FastAPI backend deployed on port 17862 (`/api/`)
- Next.js frontend deployed on port 3001 (`/react/`)
- Basic TTS/VC pages created but not connected to API
- Audio components (Uploader, Player) created
