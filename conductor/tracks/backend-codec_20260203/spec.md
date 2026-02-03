# Specification: Backend Codec Routes

**Track ID:** backend-codec_20260203
**Type:** Feature
**Created:** 2026-02-03
**Status:** Draft

## Summary

Implement Codec backend routes for DualCodec and FACodec. Create routes/codec.py with encode/decode endpoints for audio codec operations.

## Context

Amphion provides neural audio codecs (DualCodec, FACodec) that compress audio into discrete tokens and reconstruct it. These are essential components for many TTS/VC models.

## User Story

As a developer using the Amphion API, I want to encode audio to discrete tokens and decode tokens back to audio using DualCodec and FACodec so that I can integrate codec operations into my workflow.

## Acceptance Criteria

- [ ] Create models/web/api/routes/codec.py with FastAPI router
- [ ] Implement POST /api/codec/dualcodec/encode endpoint
- [ ] Implement POST /api/codec/dualcodec/decode endpoint
- [ ] Implement POST /api/codec/facodec/encode endpoint
- [ ] Implement POST /api/codec/facodec/decode endpoint
- [ ] Add ModelManager methods: codec_encode(), codec_decode()
- [ ] Support audio file upload for encode, token data for decode
- [ ] Return encoded tokens (JSON) or decoded audio file
- [ ] Register routes in main.py

## Dependencies

- Existing ModelManager class
- DualCodec and FACodec implementations
- Audio file upload validation

## Out of Scope

- Training endpoints for codecs
- Custom codec configurations
- Streaming codec operations

## Technical Notes

- DualCodec and FACodec both encode to discrete tokens
- Encode: audio -> tokens
- Decode: tokens -> audio
- Tokens can be returned as JSON array or binary
