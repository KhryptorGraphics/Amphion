# Specification: Frontend SVC Pages

**Track ID:** frontend-svc-pages_20260203
**Type:** Feature
**Created:** 2026-02-03
**Status:** Draft

## Summary

Create 4 missing SVC frontend pages: DiffComoSVC, TransformerSVC, VitsSVC, MultipleContentsSVC. Each with upload UI, progress tracking, and audio playback.

## Context

The SVC section currently only has VevoSing page. The navigation expects 5 SVC models total but only 1 page exists.

## User Story

As a user of the Amphion web interface, I want dedicated pages for each SVC model so that I can easily use DiffComoSVC, TransformerSVC, VitsSVC, and MultipleContentsSVC with proper UI and progress tracking.

## Acceptance Criteria

- [ ] Create /svc/diffcomosvc/page.tsx
- [ ] Create /svc/transformersvc/page.tsx
- [ ] Create /svc/vitssvc/page.tsx
- [ ] Create /svc/multiplecontentssvc/page.tsx
- [ ] Each page has upload UI for source and reference audio
- [ ] Each page shows progress tracking
- [ ] Each page plays back generated audio
- [ ] Update /svc/page.tsx with navigation links to all 5 models
- [ ] Match existing VevoSing page style and functionality

## Dependencies

- Backend SVC routes (Track: backend-svc-ext_20260203)
- Existing VevoSing page as reference
- API client module

## Out of Scope

- New UI components (reuse existing)
- Backend implementation

## Technical Notes

- Copy VevoSing page structure
- Update API endpoints for each model
- Add "experimental" badge for models without checkpoints
- Use existing ModelCard and FileUpload components
