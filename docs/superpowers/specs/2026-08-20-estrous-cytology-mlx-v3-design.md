# Estrous Cytology MLX-VLM v3 Design

## Purpose

This specification replaces the proposed implementation in
`/Users/moexy/Downloads/estrous_cytology_mlx_v2_spec.md` without modifying that historical
file. The existing CNN, MIL, cell-centric, and remote endpoint VLM implementations remain
independent experimental baselines.

The v3 workflow reproduces the successful visual review process: inspect the whole field and
detailed regions, describe morphology without seeing or assigning a stage, score all four stages
from that evidence, preserve uncertainty, and allow sequence context to break only genuinely
ambiguous adjacent-stage ties.

## Blinded teacher corpus

The 343 images under `dataset_split/train` must be annotated in a fresh session that has no access
to existing stage metadata. Existing labels derived from filenames, earlier models, or cycle
outputs are excluded from annotation, training, validation, and adjudication.

1. Create a SHA-256 inventory containing stable sample IDs and image paths.
2. Perform and freeze an image-only annotation pass containing primary and secondary stage,
   confidence, ordinal morphology, QC, and visible evidence.
3. Only after the first pass is hashed, expose subject/day ordering for the 141 longitudinal
   images and perform a separate sequence review.
4. Keep annotations append-only and store them separately from legacy metadata.
5. Treat these labels as teacher supervision. Keep `validate` and `test` untouched until model,
   prompt, calibrator, and temporal thresholds are frozen.

This repository does not claim that the current session produced blinded labels: stage metadata
was inspected during discovery, so a new restricted annotation context is required.

## Inference architecture

### Deterministic views

The source image is never rewritten or converted to JPEG. In memory, EXIF orientation is applied
and a five-view RGB pack is produced:

- one aspect-preserving whole-field overview, capped at 1536 pixels on its longest edge;
- four overlapping full-resolution quadrants with deterministic labels and geometry.

The input SHA-256 and view-pack version are written to every result.

### Two-pass reasoning

Pass one is explicitly stage-blind. It records `absent`, `rare`, `present`, or `dominant` abundance
for cornified squames, nucleated epithelial cells, and leukocytes, plus nuclear state, cellular
arrangement, artifacts, QC, and short image-grounded evidence. Exact percentages are prohibited.

Pass two receives the images and structured morphology and returns raw scores for diestrus,
proestrus, estrus, and metestrus. A validation-fitted temperature calibrator converts raw scores
to probabilities. Primary and secondary calls are derived from those calibrated probabilities,
not trusted from free-form model text.

Both passes require versioned JSON. Invalid output gets one deterministic repair request. A second
failure produces `ungradable`; there is no silent fallback stage.

### Temporal reconciliation

The separate reconciler receives calibrated probabilities, QC, subject ID, and day. It computes a
Viterbi path using the canonical transition matrix but may change a call only when:

- the calibrated top-two margin is at or below the frozen validation threshold;
- the image-only primary and secondary calls are adjacent in the canonical cycle;
- the sequence path selects the secondary call; and
- the sequence gain meets the frozen adjustment threshold.

Every record preserves the image-only and final call plus an adjustment flag and reason. Confident
calls and nonadjacent alternatives are never overridden.

## Model and training workflow

The initial accuracy-first zero-shot bakeoff uses:

- `mlx-community/Qwen3-VL-8B-Instruct-4bit`;
- `mlx-community/gemma-3-12b-it-4bit`;
- `mlx-community/Qwen3-VL-4B-Instruct-8bit`.

Broad single-image SFT uses EstrousBank images and stage labels only. Its templated captions and
rationales are discarded. Domain adaptation mixes teacher morphology/uncertainty supervision with
broad stage replay at ratios 1:3, 1:1, and 3:1. Select the teacher-validation winner whose broad
validation macro-F1 regresses by no more than 0.02.

Multi-image training is excluded until the upstream Qwen3-VL collation failure is resolved;
multi-view inference remains supported. The Apple-Silicon optional environment pins MLX 0.32.1
and MLX-VLM 0.6.15, the official releases verified on 2026-08-20. Model and adapter revisions must
also be frozen before a test partition is opened.

## Interfaces

```text
cycles vlm-local --input FILE_OR_DIR --model MODEL --output results.jsonl \
  [--adapter PATH] [--calibrator PATH] [--sequence-manifest CSV]

cycles vlm-prepare-sft --source estrousbank|blind-teacher \
  --input PATH --output DATASET_DIR

cycles vlm-benchmark --predictions JSONL --labels CSV --output REPORT_DIR \
  [--baseline-predictions JSONL]
```

The sequence manifest columns must be exactly:

```text
sample_id,image_path,subject_id,day
```

Result JSONL contains sample/path/hash identity, subject/day, QC, ordinal morphology, raw scores,
calibrated probabilities, image-only and sequence calls, evidence, model revision, adapter hash,
calibrator hash, prompt/schema/view versions, and dependency-lock hash.

## Review GUI

The `VLM Review` tab is an evidence-first workbench:

- left: sample/day queue with pending, accepted, corrected, ungradable, and deferred filters;
- center: zoomable whole field and four deterministic tiles;
- right: editable QC, morphology, image-only/final calls, confidence, evidence, and notes;
- conditional panel: primary-versus-secondary adjudication only for uncertain adjacent stages.

Accept, correct, ungradable, defer, previous, and next have keyboard shortcuts. Reviews append new
events containing reviewer, timestamp, source-record hash, corrected fields, and note. Model
records are never overwritten and no review triggers live training. Explicit export creates a new,
non-overwritable frozen teacher dataset with its own SHA-256 manifest.

## Evaluation gates

- The paired 95% group-bootstrap lower bound for macro-F1 improvement over the best frozen baseline
  must exceed zero.
- No species, stain, or laboratory subgroup with at least 50 samples may lose more than 0.03
  macro-F1.
- Expected calibration error must be at most 0.10 and Brier score must improve against the frozen
  uncalibrated comparison.
- Temporal reconciliation must be noninferior overall within 0.01 macro-F1 and improve the frozen
  uncertain adjacent-stage subset.
- Peak MLX allocation must remain below 36 GiB; latency and throughput are measured, not promised.
- Test partitions are opened once, after prompts, weights, calibration, and thresholds are frozen.
- Invalid outputs, corrupt images, and missing sequence metadata must fail explicitly without a
  default biological stage.

## Deliberate departures from v2

YOLO and exact cell counting are removed from the critical path because there are no validated
cell-level annotations. The v2 simulated detector, incomplete MLX training loop, fabricated
throughput estimates, fixed cell-ratio arbiter, and silent stage fallback are not carried forward.
A detector can be reconsidered only after a representative cell-level annotation and evaluation
set exists.
