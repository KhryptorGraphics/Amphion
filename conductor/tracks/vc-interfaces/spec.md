# VC Interfaces Track

## Objective

Connect the Voice Conversion frontend pages to the FastAPI backend, enabling users to convert voices using Vevo Voice, Vevo Timbre, and Vevo Style models.

## Acceptance Criteria

- [ ] Vevo Voice page allows source and reference audio upload
- [ ] Vevo Timbre page allows source and reference audio upload
- [ ] Vevo Style page allows source and reference audio upload
- [ ] All pages show real-time progress during conversion
- [ ] Generated audio plays back in AudioPlayer component
- [ ] Download button works for converted audio

## Dependencies

- FastAPI backend running on port 17862 ✅
- Audio components (Uploader, Player) created ✅
- VC page shells exist ✅
- WebSocket hook (from TTS track)

## Files to Modify

- `/var/www/aphion/frontend/src/app/vc/vevo-voice/page.tsx`
- `/var/www/aphion/frontend/src/app/vc/vevo-timbre/page.tsx`
- `/var/www/aphion/frontend/src/app/vc/vevo-style/page.tsx`
