# Model Manager Extension Plan

## Phase 1: Noro Voice Conversion

- [ ] Task 1.1: Add Noro model loader to ModelManager
- [ ] Task 1.2: Create /api/vc/noro endpoint
- [ ] Task 1.3: Test Noro inference
- [ ] Verification: Noro VC works through API

## Phase 2: SVC Support (Optional)

- [ ] Task 2.1: Add DiffComoSVC or TransformerSVC loader
- [ ] Task 2.2: Create /api/svc/ route file
- [ ] Task 2.3: Test SVC inference
- [ ] Verification: SVC works through API

## Phase 3: Additional TTS Models (Optional)

- [ ] Task 3.1: Add FastSpeech2 or VITS loader
- [ ] Task 3.2: Add endpoints to TTS routes
- [ ] Verification: Additional TTS models work

## Phase 4: Text-to-Audio (Optional)

- [ ] Task 4.1: Add AudioLDM loader
- [ ] Task 4.2: Create /api/tta/ route file
- [ ] Verification: TTA works through API

Note: Phases 2-4 are optional and depend on model availability and user needs.
