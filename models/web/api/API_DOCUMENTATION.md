# Amphion API Documentation

Complete API reference for the Amphion audio generation toolkit.

## Base URL

```
/api
```

## Authentication

All API endpoints require authentication via an API key header:

```
X-API-Key: your-api-key
```

In development mode, you can use the default key: `amphion-dev-key-change-in-production`

## Rate Limiting

API requests are rate-limited to prevent abuse. Current limits:
- 10 requests per minute per endpoint
- 100 requests per hour per API key

## Endpoints Overview

| Category | Endpoints | Description |
|----------|-----------|-------------|
| TTS | 4 | Text-to-Speech synthesis |
| VC | 4 | Voice Conversion |
| SVC | 5 | Singing Voice Conversion |
| TTA | 2 | Text-to-Audio generation |
| Codec | 4 | Audio codec encoding/decoding |
| Vocoder | 3 | Spectrogram to audio |

---

## Text-to-Speech (TTS)

### POST /api/tts/maskgct

Generate speech from text using MaskGCT (Masked Generative Codec Transformer).

**Parameters:**
- `text` (string, required): Text to synthesize
- `reference_audio` (file, required): Reference audio for voice cloning
- `language` (string, optional): Language code (default: "en")
- `temperature` (float, optional): Sampling temperature (default: 0.7)

**Response:** Audio file (WAV)

### POST /api/tts/vevo

Generate speech using Vevo TTS.

**Parameters:**
- `text` (string, required): Text to synthesize
- `reference_audio` (file, required): Reference audio for voice
- `mode` (string, optional): "fm" or "ar" (default: "fm")

### POST /api/tts/metis

Generate speech using Metis model.

**Parameters:**
- `text` (string, required): Text to synthesize
- `reference_audio` (file, optional): Reference for voice style

### POST /api/tts/dualcodec-valle

Generate speech using DualCodec VALL-E.

**Parameters:**
- `text` (string, required): Text to synthesize
- `reference_audio` (file, required): Reference audio

---

## Voice Conversion (VC)

### POST /api/vc/noro

Voice conversion using Noro model.

**Parameters:**
- `source_audio` (file, required): Source speech audio
- `target_audio` (file, required): Target voice reference

### POST /api/vc/vevo-timbre

Convert voice timbre using Vevo.

**Parameters:**
- `source_audio` (file, required): Source audio
- `reference_audio` (file, required): Timbre reference
- `flow_matching_steps` (int, optional): Number of steps (default: 32)

### POST /api/vc/vevo-style

Convert voice style using Vevo.

**Parameters:**
- `source_audio` (file, required): Source audio
- `reference_audio` (file, required): Style reference

### POST /api/vc/vevo-voice

Full voice conversion using Vevo.

**Parameters:**
- `source_audio` (file, required): Source audio
- `reference_audio` (file, required): Voice reference
- `mode` (string, optional): "fm" or "ar"

---

## Singing Voice Conversion (SVC)

### POST /api/svc/vevosing

Singing voice conversion using VevoSing.

**Parameters:**
- `content_audio` (file, required): Source singing audio (melody)
- `reference_audio` (file, required): Reference voice (timbre)
- `mode` (string, optional): "fm" (timbre only) or "ar" (full control)
- `use_shifted_src` (bool, optional): Use pitch-shifted source
- `flow_matching_steps` (int, optional): Number of steps (default: 32)

**Response:** Converted audio file (WAV)

### POST /api/svc/diffcomosvc

Singing voice conversion using DiffComoSVC (experimental).

**Parameters:**
- `content_audio` (file, required): Source singing
- `reference_audio` (file, required): Reference voice

### POST /api/svc/transformersvc

Singing voice conversion using TransformerSVC (experimental).

**Parameters:**
- `content_audio` (file, required): Source singing
- `reference_audio` (file, required): Reference voice

### POST /api/svc/vitssvc

Singing voice conversion using VitsSVC (experimental).

**Parameters:**
- `content_audio` (file, required): Source singing
- `reference_audio` (file, required): Reference voice

### POST /api/svc/multiplecontentssvc

Singing voice conversion using MultipleContentsSVC (experimental).

**Parameters:**
- `content_audio` (file, required): Source singing
- `reference_audio` (file, required): Reference voice

---

## Text-to-Audio (TTA)

### POST /api/tta/audioldm

Generate audio from text using AudioLDM (latent diffusion).

**Parameters:**
- `text` (string, required): Text description of desired audio
- `num_inference_steps` (int, optional): Diffusion steps (default: 50, range: 10-200)
- `audio_length_in_s` (float, optional): Audio duration (default: 5.0, range: 1-30)

**Example:**
```bash
curl -X POST /api/tta/audioldm \
  -H "X-API-Key: your-key" \
  -F "text=A dog barking in the park" \
  -F "num_inference_steps=50" \
  -F "audio_length_in_s=5.0"
```

**Response:** Audio file (WAV)

### POST /api/tta/picoaudio

Generate audio from text using PicoAudio (lightweight/fast).

**Parameters:**
- `text` (string, required): Text description
- `duration` (float, optional): Audio duration (default: 3.0, range: 1-10)

**Example:**
```bash
curl -X POST /api/tta/picoaudio \
  -H "X-API-Key: your-key" \
  -F "text=Sound of rain falling" \
  -F "duration=3.0"
```

---

## Audio Codecs

### POST /api/codec/dualcodec/encode

Encode audio using DualCodec.

**Parameters:**
- `audio` (file, required): Audio file to encode

**Response:**
```json
{
  "tokens": [[1, 2, 3, ...]],
  "sample_rate": 24000
}
```

### POST /api/codec/dualcodec/decode

Decode tokens to audio using DualCodec.

**Parameters:**
- `tokens` (string, required): JSON array of token indices

**Response:** Audio file (WAV)

### POST /api/codec/facodec/encode

Encode audio using FACodec (factorized codec).

**Parameters:**
- `audio` (file, required): Audio file to encode

**Response:**
```json
{
  "content_tokens": [[...]],
  "timbre_tokens": [[...]],
  "prosody_tokens": [[...]],
  "sample_rate": 24000
}
```

### POST /api/codec/facodec/decode

Decode tokens to audio using FACodec.

**Parameters:**
- `tokens` (string, required): JSON object with factorized tokens

**Response:** Audio file (WAV)

---

## Vocoders

### POST /api/vocoder/bigvgan

Vocode using BigVGAN (large-scale GAN).

**Parameters:**
- `audio` (file, required): Input audio or mel-spectrogram

**Response:** Audio file (WAV)

### POST /api/vocoder/hifigan

Vocode using HiFiGAN (high-fidelity, fast).

**Parameters:**
- `audio` (file, required): Input audio or spectrogram

**Response:** Audio file (WAV)

### POST /api/vocoder/generic

Vocode using specified vocoder backend.

**Parameters:**
- `audio` (file, required): Input audio or spectrogram
- `vocoder_name` (string, required): "bigvgan", "hifigan", or "nsf_hifigan"

**Response:** Audio file (WAV)

---

## Utility Endpoints

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### GET /api/docs

Interactive API documentation (Swagger UI).

### GET /api/openapi.json

OpenAPI specification in JSON format.

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message description"
}
```

Common HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `401` - Unauthorized (missing/invalid API key)
- `413` - Payload Too Large (file too big)
- `415` - Unsupported Media Type (invalid file format)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

---

## File Upload Limits

- Maximum file size: 100MB for content audio, 50MB for reference audio
- Supported formats: WAV, MP3, FLAC, OGG
- Recommended: WAV format for best quality

---

## WebSocket API

Real-time progress updates available via WebSocket:

```
/ws/progress/{task_id}
```

Connect to receive progress updates during long-running inference tasks.

---

## Testing

Integration tests are available in `models/web/api/tests/`:

```bash
# Run all tests
python -m pytest models/web/api/tests/ -v

# Run specific test file
python -m pytest models/web/api/tests/test_svc_routes.py -v
```

---

## Models and Research

For more information about the underlying models:

- **MaskGCT**: Masked Generative Codec Transformer for TTS (ICLR 2025)
- **Vevo**: Voice conversion with flow matching
- **DualCodec**: Dual-stream neural audio codec
- **FACodec**: Factorized audio codec for controllable generation
- **BigVGAN/HiFiGAN**: GAN-based vocoders

See the main Amphion repository for model details and training information.
