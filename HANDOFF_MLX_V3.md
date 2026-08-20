# MLX-VLM v3 Engineering Handoff

**Date:** 2026-08-20  
**Branch:** `feature/estrous-mlx-v3`  
**Isolated worktree:** `/tmp/cycles-estrous-mlx-v3`  
**Source checkout:** `/Volumes/SSD/code/cycles`  
**Status:** implementation foundation complete and now verified against real weights on Metal;
scientific annotation, model training, and held-out validation remain.

## Verified state

```text
.venv/bin/python -m pytest -q
138 passed in 5.47s

.venv/bin/python -m ruff check .
All checks passed!
```

The optional environment resolves and installs:

```text
mlx==0.32.1
mlx-vlm==0.6.15
```

**Superseded on 2026-08-20.** The original session ran in a headless sandbox with no Metal device
and could only inspect the installed 0.6.15 source, so it made no claim that inference had run. A
later session on a Metal-capable machine ran the smoke test and found that the code did not work.
Three defects were fixed in `e7e6521`:

1. `generate()` does not apply the chat template. The prompt reached the model with no image
   placeholder tokens, so `get_input_embeddings` tried to scatter a 22,097,920-element vision
   embedding into an empty position set and raised
   `[broadcast_shapes] Shapes (22097920) and (0) cannot be broadcast`. The backend now calls
   `apply_chat_template` with one placeholder per materialized view.
2. `ImagePrediction.from_dict` required `secondary_stage` to equal `ranked[1]` of the probabilities
   sorted by key. A confident model emits a one-hot distribution whose three zero-probability stages
   tie, so the occupant of that slot was an arbitrary artifact of dict ordering that no model could
   be asked to guess. Every confident prediction was rejected as `invalid_model_output`, including
   after the repair pass. Validation now compares probability values, so ties pass and genuine rank
   violations still fail.
3. The backend boundary test asserted the raw prompt reached `generate()` unchanged — encoding
   defect 1 as expected behavior, against a fake `mlx_vlm` written to match the code rather than the
   library. This is why 134 tests passed over broken inference. The fake now mirrors the real calling
   contract and fails the way the model does.

The lesson worth carrying forward: a hand-written fake of an unexecuted dependency proves only
self-consistency. Source inspection is not execution.

Verified end to end on one non-held-out training image
(`dataset_split/train/batch_1/mouse3/mouse3D1.webp`, sha256 `f53bd055…`):

```text
qc_status         usable
primary_stage     metestrus   (secondary diestrus, confidence medium)
model load         3.61 s
peak after load    4.756 GiB
inference          59.03 s / image   (five views, two passes)
peak overall       6.949 GiB         (budget 36 GiB)
```

The stage above is a model output on a training image, not a validated result, and carries no
evidentiary weight for accuracy. It shows only that the path runs and emits a schema-valid record.

Latency is worth noting before the bakeoff: 59 s/image on the smallest candidate implies roughly
5.6 hours for a single pass over the 343 teacher images, and the 8B/12B candidates will be slower.

## What is implemented

- `cycles/vlm_local/`: versioned schemas, deterministic overview/quadrant views, two-pass prompts,
  lazy MLX adapter, temperature calibration, guarded temporal reconciliation, append-only review
  events, frozen teacher export, SFT preparation, and benchmark reporting.
- CLI commands: `vlm-local`, `vlm-prepare-sft`, and `vlm-benchmark`, including an optional calibrator,
  sequence manifest, and frozen-baseline comparison.
- Evaluation: macro metrics, ECE, multiclass Brier score, species/stain/lab summaries, paired
  group-bootstrap confidence intervals, relative-improvement gate, and subgroup-regression gate.
- GUI: a `VLM Review` tab with queue filters, zoomable overview, four deterministic tiles, editable
  morphology/QC/stages/evidence, guarded transition panel, keyboard review actions, append-only logs,
  and explicit frozen export.
- Data preparation: EstrousBank tar shards become single-image MLX-VLM messages using image and stage
  only; templated captions and rationales are discarded. Frozen teacher exports retain reviewed
  morphology and uncertainty.
- Provenance: image SHA-256, model/revision, adapter hash, calibrator hash, prompt/schema/view-pack
  versions, and lockfile hash.
- Documentation: the v3 design preserves the historical Downloads specification and explicitly
  records why its simulated detector and fixed cell-ratio arbiter were not retained.

## Important boundaries

- The existing CNN, MIL, cell-centric, and remote endpoint VLM services were not replaced.
- No existing stage metadata from the local slides may be used for teacher annotation. This session
  inspected those labels and is contaminated for blind annotation.
- No model reviews update weights live. Review logs append new immutable events; SFT data appears only
  through explicit non-overwritable export.
- The source checkout contained seven unrelated untracked dataset utility scripts. They remain
  untouched in the source checkout and are not part of this feature worktree.
- No model weights were downloaded, no paid APIs were called, and no held-out partition was opened.

## Resume commands

```bash
cd /tmp/cycles-estrous-mlx-v3
env UV_CACHE_DIR=/tmp/cycles-uv-cache uv sync --extra mlx --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

The native Metal smoke test below has been run and passes; rerun it after any backend change:

```bash
cycles vlm-local --input /path/to/one/non-held-out-slide.png \
  --model mlx-community/Qwen3-VL-4B-Instruct-8bit \
  --output /tmp/cycles-mlx-smoke.jsonl
```

Do not run the test partition during this smoke test. Record model commit, peak MLX allocation,
latency, and the emitted provenance before beginning the candidate bakeoff.

## Remaining work

See [`TODO_MLX_V3.md`](TODO_MLX_V3.md). The next non-negotiable step is a fresh, stage-blind teacher
annotation session—not additional tuning against legacy labels.
