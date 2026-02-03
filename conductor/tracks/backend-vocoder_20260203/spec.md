# Specification: Backend Vocoder Routes

**Track ID:** backend-vocoder_20260203
**Type:** Feature
**Created:** 2026-02-03
**Status:** Draft

## Summary

Implement Vocoder backend routes for HiFi-GAN, BigVGAN, and other neural vocoders. Create routes/vocoder.py with inference endpoints.

## Context

Neural vocoders convert mel-spectrograms or acoustic features to raw audio. Amphion supports HiFi-GAN, BigVGAN, and others. These are used in the final stage of TTS/VC pipelines.

## User Story

As a developer using the Amphion API, I want to convert mel-spectrograms to audio using neural vocoders so that I can complete TTS/VC pipelines or generate audio from custom features.

## Acceptance Criteria

- [ ] Create models/web/api/routes/vocoder.py with FastAPI router
- [ ] Implement POST /api/vocoder/hifigan endpoint
- [ ] Implement POST /api/vocoder/bigvgan endpoint
- [ ] Add ModelManager method: vocoder_inference()
- [ ] Support mel-spectrogram or acoustic feature input
- [ ] Support vocoder selection (hifigan, bigvgan, etc.)
- [ ] Return generated audio file (WAV format)
- [ ] Register routes in main.py

## Dependencies

- Existing ModelManager class
- HiFi-GAN and BigVGAN implementations
- Pretrained vocoder checkpoints

## Out of Scope

- Training endpoints for vocoders
- Custom vocoder configurations
- Real-time streaming

## Technical Notes

- Vocoders take mel-spectrograms as input
- Input can be uploaded as numpy file or computed from audio
- Support multiple vocoder types via parameter
