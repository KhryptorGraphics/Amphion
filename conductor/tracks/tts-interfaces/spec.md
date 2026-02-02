# TTS Interfaces Track

## Objective

Connect the TTS frontend pages to the FastAPI backend, enabling users to generate speech using MaskGCT, DualCodec-VALLE, and Vevo TTS models.

## Acceptance Criteria

- [ ] MaskGCT page allows text input and reference audio upload
- [ ] DualCodec-VALLE page allows text input and reference audio upload
- [ ] Vevo TTS page allows text input and reference audio upload
- [ ] All pages show real-time progress during generation
- [ ] Generated audio plays back in AudioPlayer component
- [ ] Download button works for generated audio

## Dependencies

- FastAPI backend running on port 17862 ✅
- Audio components (Uploader, Player) created ✅
- TTS page shells exist ✅

## Files to Modify

- `/var/www/aphion/frontend/src/app/tts/maskgct/page.tsx`
- `/var/www/aphion/frontend/src/app/tts/dualcodec-valle/page.tsx`
- `/var/www/aphion/frontend/src/app/tts/vevo/page.tsx`
- `/var/www/aphion/frontend/src/hooks/useWebSocket.ts` (create)
