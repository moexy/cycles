# MLX-VLM v3 TODO

## P0 — establish valid evidence

- [ ] Start a fresh restricted annotation session with access only to
  `/Volumes/SSD/Imaging/Cycles/dataset_split/train`; deny access to all legacy stage metadata.
- [ ] Create and freeze a SHA-256 inventory of all 343 teacher images.
- [ ] Complete the image-only morphology/stage/uncertainty pass and hash the annotation log.
- [ ] Expose subject/day ordering only after that freeze and complete the second sequence pass for
  the 141 longitudinal images.
- [x] Run one native-Metal, non-held-out inference smoke test and verify the installed MLX-VLM API,
  JSON repair path, memory accounting, and provenance. Done 2026-08-20 for Qwen3-VL 4B 8-bit; it
  found three real defects, fixed in `e7e6521`. Peak 6.95 GiB, 59 s/image.
- [ ] Repeat the smoke test for the two remaining candidates (Qwen3-VL 8B 4-bit, Gemma 3 12B 4-bit)
  before the bakeoff, confirming peak allocation stays under 36 GiB at the larger sizes.
- [x] Diagnose the 59 s/image latency. The model was never the bottleneck (decode 58-74 tok/s). Three
  causes fixed in `471b536`: a repair round-trip on every confident image, no prefix reuse between
  the two passes, and an underspecified stage prompt. Now 21 s/image, ~2 h for the 343 teacher
  images.
- [ ] Decide the view-pack resolution/accuracy tradeoff before the bakeoff. The pack sends 8.81 MP
  (a 1536-edge overview plus four full-resolution quadrants) from a 2880x2048 source, which is
  8,843 tokens and ~16 s of the remaining 21 s. Halving the quadrant edge would cut that roughly
  fourfold. Whether that loses the nuclear detail separating clear from ghost nuclei is an empirical
  question; run it as an ablation rather than assuming either way.
- [ ] Fix the prefill mode for all scored runs and state it in the frozen configuration. Prefix reuse
  is self-consistent but not bit-identical to cold prefill, so a bakeoff must not mix the two.

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
