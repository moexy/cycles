# Morphology sensitivity probes

These artifacts hold the full prompts, raw model responses, strict parse results, immutable model
revisions, source-image hashes, and timings for the four controlled morphology cases.

They are **not accuracy results**. The expected stages are textbook design expectations, not
independently annotated ground truth, and the same fixed training image is reused as visual context
for counterfactual morphology. The probe tests prompt/schema sensitivity while controlling the image.

## Results

| prompt | model | valid outputs | design-expectation matches |
|---|---|---:|---:|
| v2 | Qwen3-VL 4B 8-bit | 3/4 | 0/4 |
| v2 | Qwen3-VL 8B 4-bit | 2/4 | 0/4 |
| v2 | Gemma 3 12B 4-bit | 2/4 | 0/4 |
| v3 | Qwen3-VL 4B 8-bit | 4/4 | 4/4 |
| v3 | Qwen3-VL 8B 4-bit | 4/4 | 4/4 |
| v3 | Gemma 3 12B 4-bit | 4/4 | 4/4 |

Prompt v3 adds the missing stage criteria and removes redundant model-authored probability, rank,
and confidence fields. The code derives those fields from the returned evidence scores. Those scores
are not internal logits, so probability calibration remains an empirical validation task.
