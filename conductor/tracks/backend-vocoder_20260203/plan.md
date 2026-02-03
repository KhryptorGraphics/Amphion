# Implementation Plan: Backend Vocoder Routes

**Track ID:** backend-vocoder_20260203
**Spec:** [spec.md](./spec.md)
**Created:** 2026-02-03
**Status:** [x] Complete

## Overview

Create neural vocoder backend routes supporting HiFi-GAN and BigVGAN models.

## Phase 1: Model Manager Methods

Add vocoder inference method to ModelManager.

### Tasks

- [x] Task 1.1: Add vocoder_inference() method to ModelManager
- [x] Task 1.2: Support HiFi-GAN model loading
- [x] Task 1.3: Support BigVGAN model loading

### Verification

- [x] ModelManager can load vocoder models
- [x] Inference method returns audio data

## Phase 2: Route Implementation

Create the vocoder routes file.

### Tasks

- [x] Task 2.1: Create models/web/api/routes/vocoder.py
- [x] Task 2.2: Implement POST /hifigan endpoint
- [x] Task 2.3: Implement POST /bigvgan endpoint
- [x] Task 2.4: Add mel-spectrogram input handling

### Verification

- [x] Routes file created
- [x] Endpoints accept mel-spectrogram input

## Phase 3: Integration

Register routes and test.

### Tasks

- [x] Task 3.1: Register vocoder router in main.py
- [x] Task 3.2: Test endpoints with curl

### Verification

- [x] API server starts
- [x] Endpoints respond correctly

## Final Verification

- [x] All acceptance criteria met
- [x] Code follows existing patterns
- [x] Ready for integration
