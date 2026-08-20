# MLX-VLM v3 Engineering Handoff

**Date:** 2026-08-20  
**Branch:** `feature/estrous-mlx-v3`  
**Isolated worktree:** `/tmp/cycles-estrous-mlx-v3`  
**Source checkout:** `/Volumes/SSD/code/cycles`  
**Status:** implementation foundation complete; scientific annotation, model training, and held-out
validation remain.

## Verified state

```text
.venv/bin/python -m pytest -q
134 passed in 21.88s

.venv/bin/python -m ruff check .
All checks passed!
```

The optional environment resolves and installs:

```text
mlx==0.32.1
mlx-vlm==0.6.15
```

`mlx_vlm` reaches MLX initialization, but this headless sandbox has no Metal device and raises:

```text
RuntimeError: [metal::load_device] No Metal device available.
```

Therefore no claim is made that real model inference has run here. The installed 0.6.15 source was
inspected to verify that `load()` accepts `adapter_path` and `revision`, while `generate()` accepts
image paths rather than PIL objects. The backend now materializes deterministic views as temporary
lossless PNGs and deletes them after each call; this behavior has a passing API-boundary test.

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

For a native Metal smoke test, run from a normal macOS terminal rather than the headless sandbox:

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
