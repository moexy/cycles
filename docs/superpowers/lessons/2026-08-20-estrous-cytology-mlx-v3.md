# Estrous Cytology MLX-VLM v3 Lessons

1. The successful classification behavior is morphology-first, not cell-count-first. Whole-field
   composition, detailed regions, nuclear state, arrangement, leukocyte certainty, artifacts, and
   explicit ambiguity matter more than fabricated exact percentages.
2. EstrousBank's captions and rationales are label-derived templates. They must not supervise
   morphology explanations; use only images and stage labels for the broad task.
3. Existing local metadata mixes filename-derived labels with older model/cycle outputs. It is not
   an independent reference and must remain invisible during teacher annotation.
4. Annotation contamination is irreversible within a session. Hash and freeze the image-only pass
   before exposing sequence order, and use a genuinely fresh restricted context.
5. Temporal context is useful only as a guarded tie-breaker. Preserve image-only calls and prohibit
   overrides of confident or nonadjacent alternatives.
6. Calibration must operate on raw four-stage scores using validation-fitted parameters. Numeric
   probabilities emitted by a generative model are not automatically calibrated.
7. MLX-VLM 0.6.15's `generate()` boundary takes image paths, despite the internal pipeline working
   naturally with PIL images. Temporary lossless PNG views are the compatible, auditable bridge.
8. Importing MLX-VLM initializes Metal. A headless sandbox can install and statically verify the
   package yet still cannot prove runtime inference; report that limitation explicitly.
9. Review corrections are data, not online learning events. Append immutable events tied to a model
   record hash, then export a separate frozen training corpus deliberately.
10. Relative improvement and subgroup safety are more defensible than arbitrary historical targets.
    Test data must remain sealed until prompts, adapters, calibration, and thresholds are frozen.
11. A label list is not a domain contract. The first stage prompt named all four stages but omitted
    their cytologic definitions; three different models then made the same systematic inversions.
    Explicit stage criteria fixed the controlled sensitivity probe on every candidate.
12. Do not delegate arithmetic and redundant consistency fields to a generative model. Request the
    minimal evidence scores and rationale, then derive rank, probabilities, and confidence in code.
    The returned scores are still self-reported evidence values, not internal logits.
13. Counterfactual prompt probes are design diagnostics, not accuracy estimates. Their expected
    outputs must be labeled as assumptions unless independently annotated ground truth exists.
