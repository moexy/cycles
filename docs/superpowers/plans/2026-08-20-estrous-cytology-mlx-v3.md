# Estrous Cytology MLX-VLM v3 Implementation Plan

## Goal

Deliver a local, auditable morphology-first workflow that matches expert visual review more closely
than the existing experimental pipelines, then validate it broadly before adapting to the user's
Alcian-blue slides.

## Completed foundation

- [x] Preserve the v2 specification and add a source-controlled v3 design.
- [x] Isolate the local VLM package from the endpoint VLM and frozen experimental baselines.
- [x] Implement deterministic overview plus four-quadrant preprocessing without JPEG intermediates.
- [x] Implement stage-blind morphology extraction followed by constrained four-stage scoring.
- [x] Validate JSON, retry repair once, and return ungradable rather than a default stage.
- [x] Add validation-fitted temperature calibration and complete inference provenance.
- [x] Add Viterbi reconciliation guarded by uncertainty, adjacency, runner-up, and gain thresholds.
- [x] Add append-only review events and non-overwritable teacher export.
- [x] Add EstrousBank and blind-teacher single-image SFT preparation.
- [x] Add CLI entry points, evidence-first GUI, group-bootstrap comparison, subgroup gates, and tests.
- [x] Pin and resolve MLX 0.32.1 plus MLX-VLM 0.6.15.

## Execution sequence from here

1. Produce the fresh blinded teacher corpus in two frozen passes.
2. Perform native-Metal smoke tests on non-held-out images and measure actual resource use.
3. Bake off the three candidate models using the frozen broad validation design.
4. Train the broad stage adapter, then the three replay-ratio domain adapters.
5. Fit and freeze calibration and reconciliation thresholds using validation only.
6. Open held-out partitions once and apply all relative, subgroup, calibration, temporal, and memory
   gates.
7. Promote a model only if every safety gate passes; otherwise preserve results and return to the
   preceding training decision without consulting the test data again.

## Acceptance evidence

- Automated foundation: `134 passed`; `ruff check .` clean.
- Runtime dependency state: locked and installed, but native Metal inference still required outside
  the current headless sandbox.
- Scientific completion requires the checks in `TODO_MLX_V3.md`; automated code tests alone do not
  establish model accuracy.
