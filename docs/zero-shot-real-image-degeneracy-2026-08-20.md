# Zero-shot behavior on real images

**Date:** 2026-08-20
**Prompt:** `morphology-first-v3` · **Prefill:** prefix reuse `on` · **View pack:** `overview-quadrants-v1` (default resolution)
**Images:** the same 24 stage-blind, non-held-out training images used in the view-pack resolution ablation
**Raw:** `docs/probes/zero-shot-real-image-degeneracy-2026-08-20.json`, plus per-model prediction JSONL in `docs/probes/`

## What this measures, and what it does not

This probe asks one question: **does the model's output vary with the image?** It does not measure
accuracy. No teacher labels exist for these images, so nothing here says which model is right. A
model that varies can still be systematically wrong, and a strong modal stage can reflect a real
base rate rather than a defect.

It was run because the controlled counterfactual probe cannot detect this failure. That probe
*supplies* the morphology in text and checks that pass 2 responds coherently. All three candidates
score 4/4 on it. A model whose pass 1 barely reads the image would still pass, because pass 1 is
not what it tests.

## Result

```text
                      Qwen3-VL 4B 8-bit      Qwen3-VL 8B 4-bit      Gemma 3 12B 4-bit
stage                 estrus 24              proestrus 20           proestrus 17
                                             estrus 3               estrus 3
                                             metestrus 1            metestrus 2
                                                                    diestrus 2
confidence            low 24                 low 20, medium 4       high 22, medium 2
qc_status             usable 24              usable 24              usable 24
cornified_squames     present 11             present 8, rare 8      present 11, rare 5
                      absent 13              absent 7, dominant 1   absent 8
nucleated_epithelial  present 24             present 22             dominant 17, present 6
                                             dominant 2             rare 1
leukocytes            present 17, rare 6     rare 10, absent 14     present 13, rare 10
                      absent 1                                      dominant 1
nuclear_state         mixed 24               mixed 23               clear_nuclei 20, mixed 3
                                             clear_nuclei 1         ghost_nuclei 1
arrangement           mixed 24               mixed 22, isolated 2   clusters 13, isolated 10
                                                                    mixed 1
------------------------------------------------------------------------------------------
constant fields       6 of 8                 1 of 8                 1 of 8
distinct stages       1                      3                      4
```

**The 4B is degenerate; the other two are not.** Six of the 4B's eight output fields never move
across 24 images. The 8B and 12B vary on every field except QC, which is plausibly genuinely
constant on a curated training partition.

This overturns the blocking TODO item that generalized "no candidate can stage zero-shot" from the
4B's behavior. That claim was true of one model.

## Cross-model agreement

```text
4B vs 8B      3/24   (12%)
4B vs 12B     3/24   (12%)
8B vs 12B    16/24   (67%)
all three     1/24
```

The 4B agrees with nothing because it answers `estrus` regardless. The 8B and 12B agree on two
thirds of images, which is consistent with two models reading the same slides imperfectly rather
than either behaving randomly.

## Two findings that should shape the annotation pass

**The 8B cannot call diestrus.** It predicts diestrus zero times out of 24, and the morphology pass
explains why: it scores leukocytes as `rare` or `absent` on all 24 images and never once `present`
or `dominant`. Diestrus is defined by leukocyte dominance, so a model that never registers abundant
leukocytes structurally cannot reach that stage. The 4B has the opposite bias — leukocytes `present`
on 17 of 24. The two disagree systematically on the single field that most discriminates stage.

**The 12B is confident.** It returns `high` confidence on 22 of 24 images with no accuracy evidence
behind it. High confidence is not a reason to prefer it as a teacher; a confidently wrong teacher
propagates error more efficiently than an uncertain one. Its confidence tier should be treated as
uncalibrated until labels exist.

## Consequence for the plan

The accuracy-first zero-shot bakeoff is no longer obviously measuring noise — two candidates produce
image-dependent output, so a comparison between them can carry signal. But it still cannot run
before teacher labels exist, because agreement between two unlabeled models adjudicates nothing.

The 4B should not be carried into the bakeoff as a staging candidate on this evidence. Whether it
remains viable *after* fine-tuning is a separate question this probe does not answer.
