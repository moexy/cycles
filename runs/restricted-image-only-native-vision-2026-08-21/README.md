# Restricted image-only native-vision annotation

- Source scope: `/Volumes/SSD/Imaging/Cycles/dataset_split/train`
- Inventory: 343 images, 168,804,058 bytes, no duplicate-content groups
- Frozen content-inventory SHA-256: `6a7def23bcb8640d7694840541c00ec371e0ce0dc864cd5a485d375b8aa15a4f`
- Annotation order: ascending image SHA-256
- Identity boundary: hash-only copied filenames; no subject, day, sequence, legacy stage metadata, local model predictions, or repository staging workflow used for visual decisions
- Annotator: OpenAI native visual reasoning in the active restricted image-only pass
- Interpretation: independent image-level annotations for later benchmarking; not experimentally validated ground truth and not sequence-adjudicated

`annotations.jsonl` is append-only during the pass. Each row records cytology morphology, primary and secondary stage, confidence, QC, and concise visual evidence keyed only by `image_sha256`.

## Checkpoint: 2026-08-21

- Completed: 228 / 343 images
- Remaining: 115 images
- Last completed SHA-256: `b631fd13b19f07feafdee6ba1c13c9fb9ea9dcce7c44aab64f21ee2e37b5a64b`
- Next image SHA-256: `b648b0fc97aa330cedde9a63cf8077fb5150cf4de70ae3c35490d6ac9ac0a702`
- Annotation JSONL SHA-256 at this checkpoint: `7c282449382661eee3fc0fde03dd5c807e7f9c7bd81c4b457dbd7261976e825e`
- Integrity check: 228 valid JSON records, 228 unique hashes, exactly matching the first 228 blinded images in ascending SHA-256 order

This checksum is a resumable checkpoint, not the final frozen blind-annotation hash. Continue appending in hash order; do not expose source paths, labels, subject/day identity, sequence, or local workflow outputs until all 343 image-only records have been completed and the final file has been frozen.
