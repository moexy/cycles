# MLX-VLM v3 Engineering Handoff

**Last updated:** 2026-08-20
**Branch:** `feature/estrous-mlx-v3`
**Isolated worktree:** `/private/tmp/cycles-estrous-mlx-v3`
**Source checkout:** `/Volumes/SSD/code/cycles` (still at `505e8b1`; this branch is not merged)
**Status:** the local VLM path now runs on real weights and is fast enough to use. No scientific
result exists yet, and one finding below calls the planned bakeoff into question.

---

## 1. Verified state

```text
env UV_CACHE_DIR=/tmp/cycles-uv-cache uv sync --extra mlx --extra dev   # in sync
.venv/bin/python -m pytest -q        142 passed
.venv/bin/python -m ruff check .     All checks passed!
mlx 0.32.1  |  mlx-vlm 0.6.15  |  mx.default_device() -> Device(gpu, 0)
```

Unlike the session that created this branch, the current machine has a working Metal device. Real
inference has now run end to end.

### Commits on this branch

```text
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

---

## 4. The blocking finding: zero-shot staging does not work

Pass 2 was given four textbook-unambiguous synthetic morphologies. It scored **0/4 at high
confidence**:

| morphology supplied | correct stage | returned |
|---|---|---|
| dominant anucleate cornified squames, sheets, no leukocytes | estrus | **metestrus** |
| overwhelmingly leukocytes, no cornified squames | diestrus | **estrus** (0.8) |
| clustered nucleated epithelial, clear nuclei | proestrus | **estrus** (0.8) |
| cornified + nucleated + abundant leukocytes | metestrus | **estrus** (0.7) |

This is **not** an artifact of the `morphology-first-v2` prompt written this session. Under the
original v1 prompt the same four cases return a constant `metestrus`, insensitive to the evidence
entirely. Both fail; v2 at least responds to its input.

On real slides the same degeneracy shows: `estrus / low` for all 24 images in the ablation, while the
morphology pass varied normally (`cornified_squames` 11 present / 13 absent). Pass 1 discriminates.
Pass 2 ignores it.

**Consequence for the plan.** The P1 "accuracy-first zero-shot bakeoff" assumes the candidates can
stage zero-shot. If the 8B and 12B candidates behave like the 4B, that comparison measures noise and
the adapter is doing all the work. Run the four synthetic cases against Qwen3-VL 8B 4-bit and
Gemma 3 12B 4-bit before spending the full bakeoff — four calls per model.

**Scope:** one model, the smallest candidate. This licenses no conclusion about the other two.

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

## 7. Contamination ledger

The annotation protocol requires a genuinely stage-blind teacher pass. Recording precisely what this
session was exposed to, so the next annotator can judge for themselves:

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

This session is therefore not disqualifying for stage-blind annotation on label grounds, but it has
seen subject/day ordering for part of the corpus. **The safe course remains a fresh restricted
context**, as the plan already requires.

---

## 8. Remaining work

See [`TODO_MLX_V3.md`](TODO_MLX_V3.md) for the full list. Ordered by what blocks what:

1. **Check zero-shot staging capability on the 8B and 12B candidates** (§4). Cheap, and it decides
   whether the bakeoff as designed is worth running at all.
2. **Fresh, stage-blind teacher annotation** in a restricted context with access only to
   `/Volumes/SSD/Imaging/Cycles/dataset_split/train`. Freeze a SHA-256 inventory of the 343 images,
   complete the image-only pass, hash the log, and only then expose subject/day ordering for the 141
   longitudinal images. Nothing downstream is valid without this.
3. **Smoke-test the two remaining candidates** on Metal and confirm peak allocation stays under
   36 GiB at the larger sizes.
4. Then: bakeoff -> broad SFT adapter -> replay-ratio domain adaptation -> freeze -> single held-out
   open, with all gates from the plan.

Fix the prefill mode across every scored run and state it in the frozen configuration.

---

## 9. Resume

```bash
cd /private/tmp/cycles-estrous-mlx-v3
env UV_CACHE_DIR=/tmp/cycles-uv-cache uv sync --extra mlx --extra dev
.venv/bin/python -m pytest -q          # expect 142 passed
.venv/bin/python -m ruff check .
```

Single-image inference (non-held-out only):

```bash
.venv/bin/cycles vlm-local \
  --input /Volumes/SSD/Imaging/Cycles/dataset_split/train/batch_1/mouse3/mouse3D1.webp \
  --model mlx-community/Qwen3-VL-4B-Instruct-8bit \
  --output /tmp/cycles-mlx-smoke.jsonl
# add --no-prompt-prefix-reuse to force cold prefill
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
