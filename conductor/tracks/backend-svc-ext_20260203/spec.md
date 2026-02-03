# Specification: Backend SVC Extensions

**Track ID:** backend-svc-ext_20260203
**Type:** Feature
**Created:** 2026-02-03
**Status:** Draft

## Summary

Extend SVC routes with 4 additional models: DiffComoSVC, TransformerSVC, VitsSVC, MultipleContentsSVC. Add endpoints and model manager methods.

## Context

The SVC module currently only supports VevoSing. Amphion has 4 additional SVC models that should be exposed through the web API.

## User Story

As a user of the Amphion web interface, I want to use different SVC models (DiffComoSVC, TransformerSVC, VitsSVC, MultipleContentsSVC) so that I can choose the best model for my singing voice conversion needs.

## Acceptance Criteria

- [ ] Extend models/web/api/routes/svc.py with 4 new endpoints
- [ ] Implement POST /api/svc/diffcomosvc endpoint
- [ ] Implement POST /api/svc/transformersvc endpoint
- [ ] Implement POST /api/svc/vitssvc endpoint
- [ ] Implement POST /api/svc/multiplecontentssvc endpoint
- [ ] Add ModelManager methods for each SVC model
- [ ] Each endpoint accepts source and reference audio
- [ ] Return converted audio file
- [ ] Handle models without pretrained checkpoints gracefully

## Dependencies

- Existing svc.py routes structure
- Existing ModelManager class
- SVC model implementations in Amphion

## Out of Scope

- Training endpoints for SVC models
- Real-time SVC conversion

## Technical Notes

- Some models may not have pretrained checkpoints available
- Follow the same pattern as vevosing endpoint
- Each model may have different parameters
- Consider adding "experimental" flag for models without checkpoints
