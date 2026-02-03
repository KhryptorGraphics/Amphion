# Specification: Backend TTA Routes

**Track ID:** backend-tta_20260203
**Type:** Feature
**Created:** 2026-02-03
**Status:** Draft

## Summary

Implement TTA (Text-to-Audio) backend routes for AudioLDM and PicoAudio models. Create routes/tta.py with endpoints for audio generation from text descriptions.

## Context

Amphion is an open-source audio generation toolkit. The web API currently supports TTS, VC, and SVC routes but lacks TTA (Text-to-Audio) endpoints for generating sound effects and audio from text descriptions.

## User Story

As a web application user, I want to generate audio from text descriptions using AudioLDM and PicoAudio models so that I can create sound effects and ambient audio programmatically.

## Acceptance Criteria

- [ ] Create models/web/api/routes/tta.py with FastAPI router
- [ ] Implement POST /api/tta/audioldm endpoint for AudioLDM inference
- [ ] Implement POST /api/tta/picoaudio endpoint for PicoAudio inference
- [ ] Add ModelManager methods: tta_audioldm_inference() and tta_picoaudio_inference()
- [ ] Support text prompt input and optional duration/num_inference_steps parameters
- [ ] Return generated audio file (WAV format)
- [ ] Include proper error handling and file cleanup
- [ ] Register routes in main.py

## Dependencies

- Existing ModelManager class in models/web/api/models/manager.py
- Existing authentication and rate limiting middleware
- AudioLDM and PicoAudio model implementations in Amphion

## Out of Scope

- Training endpoints for TTA models
- Batch processing for multiple prompts
- Real-time streaming generation

## Technical Notes

- Follow existing pattern from tts.py and vc.py routes
- AudioLDM uses latent diffusion for audio generation
- PicoAudio is a lightweight model for fast generation
- Both models generate from text prompts only (no reference audio needed)
