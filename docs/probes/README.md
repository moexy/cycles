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

## Full-pipeline resource smokes

Each `resource-smoke` artifact is one observed, non-held-out, two-pass prompt-v3 run with the same
training image. It stores the complete model record plus per-generation token counts/rates. These are
single-run measurements, not throughput estimates.

| model | load | two-pass inference | calls | peak after load | peak overall |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL 4B 8-bit | 4.30 s | 30.75 s | 2 | 4.76 GiB | 6.90 GiB |
| Qwen3-VL 8B 4-bit | 4.17 s | 31.77 s | 2 | 5.37 GiB | 7.68 GiB |
| Gemma 3 12B 4-bit | 5.75 s | 26.51 s | 2 | 7.01 GiB | 9.55 GiB |

All are below the 36 GiB gate in these runs. Latency is not a model ranking: it varies with view
tokenization and prefill behavior, and a prior 4B run on the same image took 54.9 seconds before the
telemetry rerun. The JSON artifacts, rather than this rounded table, are authoritative.
