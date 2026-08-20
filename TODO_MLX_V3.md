# MLX-VLM v3 TODO

## P0 — establish valid evidence

- [ ] Start a fresh restricted annotation session with access only to
  `/Volumes/SSD/Imaging/Cycles/dataset_split/train`; deny access to all legacy stage metadata.
- [x] Add and verify a blinded image-only mode before that session. `VLMReviewWorkspace` is now
  blinded by default: the queue and image heading show an opaque `Sample NNN` position instead of
  `record.sample_id`, the queue no longer shows `record.day`, the day-derived sequence call is
  hidden, and the queue is ordered by image content hash so subjects are not presented in day
  order. A toolbar checkbox reveals identity deliberately. Verified by four tests, including a
  sweep asserting that no sample ID, subject ID, image filename, or day string reaches any widget
  text while blinded.
- [x] Create and freeze a path-blind content inventory of all 343 teacher images. Artifact:
  `docs/inventories/vlm-teacher-train-content-inventory-2026-08-20.json`; aggregate SHA-256
  `6a7def23bcb8640d7694840541c00ec371e0ce0dc864cd5a485d375b8aa15a4f`; 168,804,058 bytes and no
  duplicate-content groups. A fresh restricted context must recompute and match it before annotation.
- [ ] Complete the image-only morphology/stage/uncertainty pass and hash the annotation log.
- [ ] Expose subject/day ordering only after that freeze and complete the second sequence pass for
  the 141 longitudinal images.
- [x] Run one native-Metal, non-held-out inference smoke test and verify the installed MLX-VLM API,
  JSON repair path, memory accounting, and provenance. Done 2026-08-20 for Qwen3-VL 4B 8-bit; it
  found three real defects, fixed in `e7e6521`. Peak 6.95 GiB, 59 s/image.
- [x] Repeat full two-pass prompt-v3 resource smokes for all candidates on the same non-held-out
  image, with immutable revisions and per-generation telemetry saved under `docs/probes/`. Observed
  peak overall: Qwen3-VL 4B 8-bit 6.90 GiB, Qwen3-VL 8B 4-bit 7.68 GiB, Gemma 3 12B 4-bit 9.55 GiB.
  All are below 36 GiB. Single-run inference times were 30.75 s, 31.77 s, and 26.51 s respectively;
  do not treat these as a stable latency ranking.
- [x] Diagnose the 59 s/image latency. The model was never the bottleneck (decode 58-74 tok/s). Three
  causes fixed in `471b536`: a repair round-trip on every confident image, no prefix reuse between
  the two passes, and an underspecified stage prompt. Now 21 s/image, ~2 h for the 343 teacher
  images.
- [x] Measure the view-pack resolution tradeoff. Halving the quadrant edge to 792 px is 1.82x faster
  and 2.02x cheaper in tokens (24.6 s -> 13.5 s median over 24 training images, `quadrant_max_edge`
  in `build_view_pack`, raw results in `docs/ablation-quadrant-resolution-2026-08-20.json`). Stage
  agreement was 24/24 but is uninformative because the model was a constant predictor; the morphology
  pass changed cornified_squames on 3/24 and leukocytes on 4/24 images.
- [ ] Re-run the resolution ablation as an accuracy comparison once teacher labels exist. Agreement
  cannot adjudicate which resolution is right; do not lower resolution before then.
- [x] Establish controlled morphology-to-stage sensitivity on all candidates. The original v2 probe
  was misreported as accuracy even though its expected stages were design expectations and the fixed
  image was not ground truth. It nevertheless exposed a real prompt/schema defect: v2 gave 0/4
  expectation matches on every candidate and only 7/12 schema-valid responses overall. Prompt v3
  defines the cytology criteria and asks only for relative evidence scores plus rationale; Qwen3-VL
  4B, Qwen3-VL 8B, and Gemma 3 12B each produce 4/4 valid, 4/4 matching controlled outputs. Raw
  artifacts are in `docs/probes/`. This proves contract compliance, not real-image accuracy.
- [ ] Rerun a small labeled, non-held-out real-image comparison under prompt v3 before committing to
  the full bakeoff. Do not infer accuracy from the controlled counterfactual probe.
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
  The VLM's `raw_scores` are self-reported evidence values, not internal logits; validate empirically
  that temperature scaling improves held-in validation NLL/Brier before calling them calibrated.
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
- [ ] Decide and validate how temporal reconciliation handles repeated observations and irregular
  day gaps; the current transition matrix advances once per record, not once per elapsed day.
- [ ] Reassess multi-image SFT only after the upstream Qwen3-VL collation issue is demonstrably fixed.
- [ ] Reconsider cell detection only after representative cell-level annotations exist.
