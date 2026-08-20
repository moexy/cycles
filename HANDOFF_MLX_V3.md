# MLX-VLM v3 Engineering Handoff

**Last updated:** 2026-08-20
**Branch:** `feature/estrous-mlx-v3`
**Isolated worktree:** `/private/tmp/cycles-estrous-mlx-v3`
**Source checkout:** `/Volumes/SSD/code/cycles` (still at `505e8b1`; this branch is not merged)
**Status:** the local VLM path runs on real weights and the morphology-to-stage contract now passes
all controlled sensitivity cases on all three candidates. This is an engineering result, not a
scientific accuracy result; no independently labeled evaluation exists yet.

---

## 1. Verified state

```text
env UV_CACHE_DIR=/tmp/cycles-uv-cache uv sync --extra mlx --extra dev   # in sync
.venv/bin/python -m pytest -q        179 passed
.venv/bin/python -m ruff check .     All checks passed!
mlx 0.32.1  |  mlx-vlm 0.6.15  |  mx.default_device() -> Device(gpu, 0)
```

Unlike the session that created this branch, the current machine has a working Metal device. Real
inference has now run end to end.

### Commits on this branch

```text
c435290 and later entries below are historical commits; see `git log main..HEAD` for the live list.
d7ae396  docs: add CLAUDE.md recording the Markdown-first and local-first conventions
ac72ab6  docs: rewrite the v3 handoff as a comprehensive session record
c435290  docs: raise zero-shot staging capability as a blocker ahead of the bakeoff
3cf106b  feat(views): add quadrant_max_edge and record the zero-shot staging ablation
7b26154  docs: record the latency diagnosis and the remaining view-pack tradeoff
471b536  perf(vlm-local): cut per-image latency from 62s to 21s and record the prefill mode
de5f310  docs: record the native-Metal smoke test result and correct the v3 handoff
e7e6521  fix(vlm-local): make MLX-VLM inference work against real weights
560085a  feat: add morphology-first local MLX-VLM workflow   <- previous session ended here
```

---

## 2. What the first real inference run exposed

The previous session ran headless with no Metal device, verified the MLX-VLM API by reading the
installed source, and correctly declined to claim inference had run. When it was finally run, the
code did not work. Three defects, fixed in `e7e6521`:

1. **The chat template was never applied.** `generate()` does not apply it; the caller must. The
   prompt reached the model carrying no image placeholder tokens, so `get_input_embeddings` tried to
   scatter a 22,097,920-element vision embedding into an empty position set:
   `[broadcast_shapes] Shapes (22097920) and (0) cannot be broadcast`.
2. **The schema rejected every confident prediction.** `ImagePrediction.from_dict` required
   `secondary_stage` to equal `ranked[1]` of the probabilities sorted by key. A confident model emits
   a one-hot distribution whose three zero-probability stages tie, so that slot's occupant was an
   arbitrary artifact of dict ordering that no model could guess. Validation now compares probability
   *values*, so ties pass and genuine rank violations still fail.
3. **The boundary test asserted the first defect as correct behavior.** `tests/test_vlm_backend.py`
   asserted the raw prompt arrived at `generate()` unchanged, against a fake `mlx_vlm` hand-written
   to match the code rather than the library. This is why 134 tests passed over inference that could
   never work.

> **Carry this forward.** A hand-written fake of a dependency you have never executed proves only
> self-consistency. Source inspection is not execution. The fake now mirrors the real calling
> contract and fails the way the model does.

---

## 3. Performance: 62 s -> 21 s per image

The first working run took 59 s/image, which was too slow to believe. Profiling showed the model was
never the bottleneck — decode runs at 58-74 tok/s, normal for a 4B 8-bit model on this hardware. The
time went to re-encoding an 8,843-token view pack up to three times per image. Fixed in `471b536`:

| cause | effect |
|---|---|
| A confident model names the same stage for primary and secondary, meaning "no distinct runner-up". The schema called that malformed and spent a full repair round-trip. | ~21 s per confident image, re-encoding the view pack to learn nothing. Now normalized to `None`. |
| Both passes send an identical view pack and differ only in the trailing instruction, yet each paid full cold prefill. | `PromptCacheState` threaded through the backend, keyed on view digests so no state crosses slides. Pass-2 prefill 539 -> ~17,800 tok/s. |
| The stage prompt never enumerated allowed values, so the model invented `"unknown"` with all-zero probabilities. | Another repair pass. Prompt bumped to `morphology-first-v2`. |

```text
before   62.2 s / image   3 generate calls
after    21.2 s / image   2 generate calls
```

A pass over the 343 teacher images is now roughly 2 h per model rather than 5.6 h.

**Checked and discarded:** `VisionFeatureCache` looks like the obvious fix, but Qwen3-VL does not
implement `encode_image`, so it silently no-ops for this model. It would have looked wired up and
done nothing.

### Prefill mode is now part of provenance

Reusing the KV prefix is a **different, self-consistent numeric path**. Both modes are deterministic —
repeated runs agree bit for bit — but reused and cold prefill can disagree on knife-edge images where
the model sits near-uniform across stages. On the smoke-test image, warm gave `estrus 0.378` and cold
`estrus 0.289`, with different runner-ups.

Records therefore carry `provenance.prompt_prefix_reuse` (`on`/`off`), and `--no-prompt-prefix-reuse`
forces cold prefill. **Scored runs must not mix the two modes.**

### Measured resource use (Qwen3-VL 4B 8-bit, five views, two passes)

```text
model load        3.61 s
peak after load   4.756 GiB
peak overall      6.949 GiB        budget 36 GiB — comfortable
inference        ~21 s / image
```

That table is the historical v2 optimization run. Direct prompt-v3 resource smokes were later run
for all candidates on the same non-held-out training image, with full telemetry in `docs/probes/`:

| model | immutable revision | load | two-pass inference | calls | peak overall |
|---|---|---:|---:|---:|---:|
| Qwen3-VL 4B 8-bit | `0943db6e…` | 4.30 s | 30.75 s | 2 | 6.90 GiB |
| Qwen3-VL 8B 4-bit | `defcdea7…` | 4.17 s | 31.77 s | 2 | 7.68 GiB |
| Gemma 3 12B 4-bit | `86cc6a8d…` | 5.75 s | 26.51 s | 2 | 9.55 GiB |

All observed peaks are below the 36 GiB gate. These are single runs, not throughput estimates or a
latency ranking. A preceding uninstrumented 4B run took 54.9 s; whether that difference was runtime
variance or an extra repair call is unknown, so it must not be silently averaged away. The final
telemetry runs each used exactly two generation calls.

---

## 4. Controlled staging probe: the blocker was the contract, not model size

The prior handoff described four counterfactual morphologies as a **0/4 zero-shot accuracy result**.
That phrasing was wrong. The fixed training image is not ground truth for the supplied morphology,
and the expected stages are textbook design expectations rather than independent annotations. The
probe measures whether pass 2 responds coherently to controlled morphology, not biological accuracy.

The underlying failure was still real. Under `morphology-first-v2`, Qwen3-VL 4B matched 0/4 design
expectations and produced only 3/4 schema-valid responses:

| morphology supplied | correct stage | returned |
|---|---|---|
| dominant anucleate cornified squames, sheets, no leukocytes | estrus | **metestrus** |
| overwhelmingly leukocytes, no cornified squames | diestrus | **estrus** (0.8) |
| clustered nucleated epithelial, clear nuclei | proestrus | **estrus** (0.8) |
| cornified + nucleated + abundant leukocytes | metestrus | **estrus** (0.7) |

Two design flaws explained the result:

1. The stage prompt named the four labels but never defined their cytologic criteria. The models
   repeatedly inverted estrus/metestrus and leukocyte-dominant diestrus.
2. The model was asked to emit raw scores, probabilities, primary/secondary labels, and confidence.
   Those are redundant arithmetic/consistency tasks. Across the first three candidate probes, only
   7/12 outputs passed strict schema validation.

`morphology-first-v3` now provides the explicit criteria and asks the model only for four finite
relative evidence scores plus a rationale. The pipeline derives probabilities, rank, and confidence
deterministically. Raw scores are model-authored evidence values, **not internal logits**; identity-
temperature softmax is an engineering default, not validated calibration.

All three immutable candidate revisions then passed the controlled probe:

| candidate | revision | v2 valid / match | v3 valid / match | v3 total generation time |
|---|---|---:|---:|---:|
| Qwen3-VL 4B 8-bit | `0943db6e…` | 3/4 / 0/4 | **4/4 / 4/4** | 28.6 s |
| Qwen3-VL 8B 4-bit | `defcdea7…` | 2/4 / 0/4 | **4/4 / 4/4** | 41.8 s |
| Gemma 3 12B 4-bit | `86cc6a8d…` | 2/4 / 0/4 | **4/4 / 4/4** | 49.2 s |

Raw prompts, responses, parses, hashes, revisions, and timings are under `docs/probes/`. The first
call includes cold view prefill and later calls reuse the shared image prefix, so totals are useful
within this probe but are not a general throughput benchmark.

On real slides under v2 the same degeneracy showed: `estrus / low` for all 24 images in the ablation, while the
morphology pass varied normally (`cornified_squames` 11 present / 13 absent). Pass 1 discriminates.
That real-image ablation has not been rerun under v3 and has no independent labels. It cannot establish
accuracy or model selection. The bakeoff remains meaningful only after a clean teacher reference exists.

---

## 5. View-pack resolution ablation

24 stage-blind training images, full-resolution quadrants vs a halved 792 px edge. Raw results in
`docs/ablation-quadrant-resolution-2026-08-20.json`.

```text
full   median 24.6 s   7,747 prompt tokens
half   median 13.5 s   3,839 prompt tokens     1.82x faster, 2.02x fewer tokens
stage agreement                                24/24
```

**The 24/24 is not evidence that resolution is free.** The model returned the same stage for every
image in both conditions, so agreement is trivially total for a predictor that ignores its input.

The informative comparison is the morphology pass, which does vary. Halving the edge changed
`cornified_squames` on 3/24 images and `leukocytes` on 4/24 — the two fields that discriminate stage.
Which call is correct cannot be settled without labels.

`build_view_pack(..., quadrant_max_edge=N)` exists and is tested. **The default is unchanged.** Do not
lower resolution until teacher labels allow this to be re-run as an accuracy comparison.

### 5.1 The constant predictor is the 4B, not the family

The ablation above left open whether any candidate could stage zero-shot at all. That has now been
tested directly on the same 24 images under prompt v3, and the degeneracy does not generalize. Full
report: [`docs/zero-shot-real-image-degeneracy-2026-08-20.md`](zero-shot-real-image-degeneracy-2026-08-20.md).

```text
                      4B 8-bit          8B 4-bit           12B 4-bit
distinct stages       1 (estrus 24)     3 (proestrus 20)   4 (proestrus 17)
constant fields       6 of 8            1 of 8             1 of 8
agreement             4B/8B 3/24        4B/12B 3/24        8B/12B 16/24
```

This probe measures **output variation, not accuracy**. No labels exist for these images, so it
cannot say which model is right. It was needed because the controlled counterfactual probe supplies
the morphology in text and therefore cannot detect a pass 1 that ignores the image — a model can
score 4/4 there and still be a constant predictor on real slides, which is exactly what the 4B is.

Two findings carry into the annotation pass:

- **The 8B cannot reach diestrus.** It scores leukocytes `rare` or `absent` on all 24 images, never
  `present`/`dominant`, and predicts diestrus 0/24. The 4B has the opposite bias (`present` 17/24).
  The candidates disagree systematically on the field that most discriminates stage.
- **The 12B confidence tier is uncalibrated.** `high` on 22/24 with no accuracy evidence behind it.

---

## 6. Why the pipeline uses two passes

Asked and answered this session, since the cost looked suspicious. The split is load-bearing, not an
efficiency artifact:

- **Anti-rationalization.** Pass 1 is instructed *"Do not assign an estrous stage."* A single combined
  call would let the model choose a stage and then write morphology justifying it — invisible in the
  output, and it would silently corrupt the evidence record. Splitting makes pass 1 structurally
  incapable of it.
- **Auditability.** Morphology is reviewed and edited independently in the GUI and is what feeds
  teacher/SFT export.
- **It mirrors expert review**, per `docs/superpowers/lessons/2026-08-20-estrous-cytology-mlx-v3.md`.

With prefix reuse, pass 2 now costs ~4 s, so the property is cheap.

**Untested lever:** pass 2 re-sends the images as well as the morphology JSON. Dropping them makes
staging a pure function of the recorded evidence (~1 s, fully auditable) at the cost of cues the
morphology schema does not capture. Worth an ablation once labels exist.

---

## 7. Later audit hardening

A later review reproduced and fixed additional paths that could silently corrupt evidence:

- A valid morphology record was discarded when stage JSON failed twice; stage failure now leaves
  morphology intact and marks only the stage prediction ungradable.
- Repair calls were stateless but referred to a "previously requested schema" that the model could
  not see; the full original request is now included.
- Ungradable records could be written but not read back; schema round-trip now supports the explicit
  no-stage/no-score representation.
- NaN/infinite scores, non-unit probabilities, duplicate IDs, and partial benchmark coverage could
  reach metrics or be silently normalized/overwritten; all are now rejected.
- Prompt-cache identity omitted image dimensions and mode; equal bytes in differently shaped images
  can no longer reuse the wrong KV state.
- Model provenance defaulted to `unspecified` and the lock hash depended on launch directory.
  `--model-revision` now requires an immutable 40-character commit SHA, and `uv.lock` is resolved
  relative to the installed repository source.
- Annotation corrections and reloaded events were under-validated, and changing a primary stage to
  the old secondary could export identical primary/secondary labels. Events are revalidated and a
  duplicate runner-up is cleared.
- Sanitized teacher sample IDs could overwrite image files. Safe filename collisions now fail and
  roll back the prepared dataset.

These are engineering validations. They do not substitute for an independent teacher reference.

### 7.1 Blinded review mode

The review workspace previously displayed `record.sample_id` in both the queue and the image
heading, and `record.day` in the queue, which would have leaked subject and day into a supposedly
stage-blind annotation pass. `VLMReviewWorkspace` now takes `blinded: bool = True` and defaults to
blinded, so an annotator who never touches the toggle is still protected.

While blinded:

- the queue row and image heading read `Sample NNN`, an opaque display position;
- `record.day` is not rendered anywhere;
- the sequence-derived **Final call** row is hidden, because it is computed from day ordering and
  would reintroduce the sequence the protocol defers;
- the queue is ordered by `image_sha256`, not by acquisition order, so consecutive days of one
  subject are not presented adjacently. The ordering is deterministic across runs.

Identity is revealed only by an explicit toolbar checkbox, and the toolbar states the current mode.
Blinding is a display concern only: `AnnotationStore` still logs the true `sample_id`, so the
review log remains fully traceable.

Four tests in `tests/test_vlm_review_gui.py` cover this, including
`test_blinded_workspace_leaks_no_identifier_into_any_widget_text`, which walks every child widget
and asserts that no sample ID, subject ID, image filename, or day string appears in any text,
tooltip, placeholder, or queue row while blinded.

## 8. Contamination ledger

The annotation protocol requires a genuinely stage-blind teacher pass. The historical claims below
were true only of the branch-building session. A later audit session read aggregate information from
`ground_truth_metadata.csv` and enumerated held-out paths while checking repository state. No held-out
image was rendered or passed to a model, but this context is contaminated for fresh annotation.

Historical branch-building session:

- **Never read:** any stage label, `manifest.csv`, `summary.json`, or any legacy stage metadata.
- **Never opened:** the `test` or `validate` partitions.
- **Never rendered into context:** any slide image. Images were passed to the model as file paths;
  no image was viewed.
- **Seen:** training-partition *paths* and filenames. Note that `batch_N/mouseN/mouseNDk.webp`
  exposes subject and day, which the protocol says stays hidden until the first annotation pass is
  frozen. The `Histology staging images` / `BulkRNA staging images` trees use obfuscated
  `sample_<hex>.webp` names.
- **Seen:** model *predictions* on ~30 training images — all degenerate `estrus`, carrying no label
  information.

The current audit session **must not perform teacher annotation**. Start a fresh restricted context
with no metadata or held-out access, as the plan already requires.

---

## 9. Remaining work

See [`TODO_MLX_V3.md`](TODO_MLX_V3.md) for the full list. Ordered by what blocks what:

1. **Fresh, stage-blind teacher annotation** in a restricted context with access only to
   `/Volumes/SSD/Imaging/Cycles/dataset_split/train`. Before displaying an image, recompute the
   path-blind 343-image inventory and match aggregate SHA-256
   `6a7def23bcb8640d7694840541c00ec371e0ce0dc864cd5a485d375b8aa15a4f`. Then complete the image-only
   pass, hash the log, and only then expose subject/day ordering for the 141 longitudinal images.
   The review UI is now blinded by default (section 7.1); confirm `workspace.blinded` is `True`
   and that the toolbar reads `Blinded` before the first image is displayed.
2. **Rerun a small labeled, non-held-out real-image comparison under prompt v3** before a full
   bakeoff. The controlled probe proves contract compliance only.
3. **Freeze the prefill mode and measurement protocol** before any scored multi-model run; the
   resource smokes establish feasibility but not comparative throughput.
4. Then: bakeoff -> decide whether SFT is warranted -> replay-ratio domain adaptation if warranted -> freeze -> single held-out
   open, with all gates from the plan.

Fix the prefill mode across every scored run and state it in the frozen configuration.

---

## 10. Resume

```bash
cd /private/tmp/cycles-estrous-mlx-v3
env UV_CACHE_DIR=/tmp/cycles-uv-cache uv sync --extra mlx --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
```

Single-image inference (non-held-out only):

```bash
.venv/bin/cycles vlm-local \
  --input /Volumes/SSD/Imaging/Cycles/dataset_split/train/batch_1/mouse3/mouse3D1.webp \
  --model mlx-community/Qwen3-VL-4B-Instruct-8bit \
  --model-revision 0943db6e15185b86be368d3cf0704aec740b142b \
  --output /tmp/cycles-mlx-smoke.jsonl
# add --no-prompt-prefix-reuse to force cold prefill
```

Verify the frozen, path-blind teacher corpus before a fresh annotation pass:

```bash
.venv/bin/python scripts/freeze_vlm_inventory.py \
  --input /Volumes/SSD/Imaging/Cycles/dataset_split/train \
  --output /tmp/vlm-teacher-inventory-check.json
cmp /tmp/vlm-teacher-inventory-check.json \
  docs/inventories/vlm-teacher-train-content-inventory-2026-08-20.json
```

Model weights are already in `~/.cache/huggingface/hub`; no re-download needed for the 4B candidate.

---

## 10. Boundaries preserved

- The existing CNN, MIL, cell-centric, and remote-endpoint VLM services were not touched.
- No held-out partition has been opened.
- No model weights were trained; no adapter exists yet.
- No paid APIs were called.
- The seven untracked dataset utility scripts in the source checkout remain untouched and are not
  part of this branch.
