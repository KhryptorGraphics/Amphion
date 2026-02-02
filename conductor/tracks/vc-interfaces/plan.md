# VC Interfaces Implementation Plan

## Phase 1: Vevo Voice Page [COMPLETE]

- [x] Task 1.1: Add state for source audio, reference audio, loading, result
- [x] Task 1.2: Add dual AudioUploader components (source + reference)
- [x] Task 1.3: Implement convert button with API call
- [x] Task 1.4: Add progress bar with WebSocket updates
- [x] Task 1.5: Integrate AudioPlayer for converted audio
- [x] Verification: Vevo Voice conversion works end-to-end

## Phase 2: Vevo Timbre Page [COMPLETE]

- [x] Task 2.1: Copy Vevo Voice pattern
- [x] Task 2.2: Adjust labels for timbre conversion context
- [x] Task 2.3: Add visual flow indicator (Style+Timbre A → Timbre B → Style+Timbre B)
- [x] Task 2.4: Add info banner explaining timbre transfer
- [x] Verification: Vevo Timbre conversion works

## Phase 3: Vevo Style Page [COMPLETE]

- [x] Task 3.1: Copy pattern for Vevo Style (accent conversion)
- [x] Task 3.2: Add style/accent-specific labels
- [x] Task 3.3: Add visual flow indicator (Style A+Timbre → Style B → Style B+Timbre)
- [x] Task 3.4: Add info banner explaining style/accent transfer
- [x] Verification: Vevo Style conversion works

## Phase 4: Polish [IN PROGRESS]

- [x] Task 4.1: Add consistent color themes (Voice=blue, Timbre=orange, Style=pink)
- [x] Task 4.2: Add error handling and user feedback
- [~] Task 4.3: Add loading states and progress indicators
- [ ] Task 4.4: Add comparison view (source vs converted)
- [ ] Verification: All VC pages handle edge cases
