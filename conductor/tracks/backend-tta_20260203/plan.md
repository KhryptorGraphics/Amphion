# Implementation Plan: Backend TTA Routes

**Track ID:** backend-tta_20260203
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-03
**Status:** [x] Complete

## Overview

Create TTA (Text-to-Audio) backend routes supporting AudioLDM and PicoAudio models. This involves creating the route file, adding ModelManager inference methods, and registering routes.

## Phase 1: Model Manager Methods

Add inference methods to ModelManager for TTA models.

### Tasks

- [x] Task 1.1: Add tta_audioldm_inference() method to ModelManager
- [x] Task 1.2: Add tta_picoaudio_inference() method to ModelManager
- [x] Task 1.3: Implement proper model loading and caching logic

### Verification

- [x] ModelManager can load TTA models without errors
- [x] Inference methods return audio data

## Phase 2: Route Implementation

Create the TTA routes file with endpoints.

### Tasks

- [x] Task 2.1: Create models/web/api/routes/tta.py
- [x] Task 2.2: Implement POST /audioldm endpoint with text prompt parameter
- [x] Task 2.3: Implement POST /picoaudio endpoint with text prompt parameter
- [x] Task 2.4: Add optional parameters (duration, num_inference_steps)
- [x] Task 2.5: Add proper error handling and file cleanup

### Verification

- [x] Routes file created with all endpoints
- [x] Endpoints accept correct parameters

## Phase 3: Integration

Register routes and verify full flow.

### Tasks

- [x] Task 3.1: Register tta router in main.py
- [x] Task 3.2: Add TTA tags to API documentation
- [x] Task 3.3: Test endpoints with curl

### Verification

- [x] API server starts without errors
- [x] Endpoints respond to requests
- [x] Audio files are generated correctly

## Final Verification

- [x] All acceptance criteria met
- [x] Code follows existing patterns
- [x] Error handling works correctly
- [x] Ready for integration
