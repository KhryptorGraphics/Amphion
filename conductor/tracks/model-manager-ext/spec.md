# Model Manager Extension Track

## Objective

Extend the ModelManager to support additional Amphion models beyond the initial four (MaskGCT, DualCodec-VALLE, Vevo TTS, Vevo VC).

## Models to Add

### Voice Conversion
- Noro (noise-robust VC)

### Singing Voice Conversion
- DiffComoSVC
- TransformerSVC
- VitsSVC

### Text-to-Audio
- AudioLDM
- PicoAudio

### Additional TTS
- FastSpeech2
- VITS
- VALL-E
- NaturalSpeech2
- Jets
- Metis

## Acceptance Criteria

- [ ] Noro model loads and runs inference
- [ ] At least one SVC model loads and runs inference
- [ ] API endpoints added for new model categories
- [ ] Model status endpoint reflects all available models

## Dependencies

- Model weights downloaded to appropriate locations
- CUDA environment properly configured

## Files to Modify

- `/home/kp/repo2/Amphion/models/web/api/models/manager.py`
- `/home/kp/repo2/Amphion/models/web/api/routes/` (new route files)
