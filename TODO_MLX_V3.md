# MLX-VLM v3 TODO

## P0 — establish valid evidence

- [ ] Start a fresh restricted annotation session with access only to
  `/Volumes/SSD/Imaging/Cycles/dataset_split/train`; deny access to all legacy stage metadata.
- [ ] Create and freeze a SHA-256 inventory of all 343 teacher images.
- [ ] Complete the image-only morphology/stage/uncertainty pass and hash the annotation log.
- [ ] Expose subject/day ordering only after that freeze and complete the second sequence pass for
  the 141 longitudinal images.
- [ ] Run one native-Metal, non-held-out inference smoke test for each candidate model and verify the
  installed MLX-VLM API, JSON repair path, memory accounting, and provenance.

## P1 — select and train

- [ ] Run the accuracy-first zero-shot bakeoff for Qwen3-VL 8B 4-bit, Gemma 3 12B 4-bit, and
  Qwen3-VL 4B 8-bit using group-bootstrap comparisons.
- [ ] Prepare broad EstrousBank SFT data and confirm templated captions/reasoning are absent.
- [ ] Train the broad single-image stage adapter; retain training logs, exact model revision, and
  adapter hash.
- [ ] Domain-adapt with teacher:broad replay ratios 1:3, 1:1, and 3:1; reject checkpoints with more
  than 0.02 broad-validation macro-F1 regression.
- [ ] Fit temperature calibration and temporal margin/gain thresholds on validation data only.
- [ ] Upgrade the transition adjudication panel from call summaries to true side-by-side neighboring
  image comparison if reviewers need it during the blinded sequence pass.

## P1 — freeze and evaluate

- [ ] Freeze model, adapter, prompt, view pack, calibrator, reconciliation thresholds, and lockfile.
- [ ] Open each held-out partition once.
- [ ] Confirm the paired group-bootstrap macro-F1 lower bound exceeds zero versus the best frozen
  baseline.
- [ ] Confirm no species/stain/lab subgroup with at least 50 samples loses more than 0.03 macro-F1.
- [ ] Compare calibrated versus uncalibrated Brier score and require ECE no greater than 0.10.
- [ ] Confirm temporal reconciliation is noninferior within 0.01 overall and improves the frozen
  uncertain adjacent-stage subset.
- [ ] Measure peak MLX allocation below 36 GiB and report latency/throughput without extrapolation.

## P2 — operational hardening

- [ ] Add a dedicated CLI command to fit and freeze a calibrator from validation raw scores.
- [ ] Add temporal subset and calibrated-versus-uncalibrated gates directly to `vlm-benchmark`.
- [ ] Add interruption-safe resume/checkpoint behavior for large local inference folders.
- [ ] Reassess multi-image SFT only after the upstream Qwen3-VL collation issue is demonstrably fixed.
- [ ] Reconsider cell detection only after representative cell-level annotations exist.
