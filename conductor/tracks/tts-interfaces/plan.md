# TTS Interfaces Implementation Plan

## Phase 1: WebSocket Hook [COMPLETE]

- [x] Task 1.1: Create useWebSocket hook for progress updates
- [x] Task 1.2: Create useProgress hook for task state management
- [x] Verification: Hook connects to WebSocket and receives messages

## Phase 2: MaskGCT Page [COMPLETE]

- [x] Task 2.1: Add state management (text, audio file, loading, result)
- [x] Task 2.2: Integrate AudioUploader for reference audio
- [x] Task 2.3: Add text input with language detection
- [x] Task 2.4: Implement generate button with API call
- [x] Task 2.5: Add progress bar with WebSocket updates
- [x] Task 2.6: Integrate AudioPlayer for generated audio
- [x] Verification: Full MaskGCT TTS workflow works end-to-end

## Phase 3: DualCodec-VALLE Page [COMPLETE]

- [x] Task 3.1: Copy MaskGCT pattern with VALLE-specific options
- [x] Task 3.2: Add VALLE-specific parameters (temperature, top_k, top_p, repeat_penalty)
- [x] Task 3.3: Add reference text input for better quality
- [x] Task 3.4: Add collapsible advanced settings panel
- [x] Verification: DualCodec-VALLE page complete with advanced settings

## Phase 4: Vevo TTS Page [COMPLETE]

- [x] Task 4.1: Copy pattern for Vevo TTS
- [x] Task 4.2: Add style/timbre reference options (separate upload support)
- [x] Task 4.3: Add language selection dropdowns
- [x] Task 4.4: Add reference audio transcript input
- [x] Verification: Vevo TTS page complete with disentangled timbre/style

## Phase 5: Polish [COMPLETE]

- [x] Task 5.1: Add loading states and error handling
- [x] Task 5.2: Add consistent progress bar styling
- [x] Task 5.3: Add model info cards with usage tips
- [x] Verification: All edge cases handled gracefully
