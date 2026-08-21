# Restricted image-only annotation handoff

**Date:** 2026-08-21

**Status:** paused cleanly at 228 / 343 images

**Artifact root:** `/Volumes/SSD/code/cycles/runs/restricted-image-only-native-vision-2026-08-21`

## What is complete

- `content-inventory.json` freezes the 343-image source inventory.
- `blinded_images/` contains hash-only copies used for direct visual inspection.
- `annotations.jsonl` contains 228 append-only morphology and provisional stage records.
- The saved records parse as JSON, contain 228 unique hashes, and match the first 228 blinded images in ascending SHA-256 order.
- Current annotation checksum: `7c282449382661eee3fc0fde03dd5c807e7f9c7bd81c4b457dbd7261976e825e`.

## Exact resume point

- Last completed: `b631fd13b19f07feafdee6ba1c13c9fb9ea9dcce7c44aab64f21ee2e37b5a64b`
- Next image: `blinded_images/b648b0fc97aa330cedde9a63cf8077fb5150cf4de70ae3c35490d6ac9ac0a702.webp`
- Remaining: 115 images, indices 229 through 343 in the sorted blinded inventory.

To list the next batch without revealing source identity:

```bash
/opt/homebrew/bin/rg --files \
  runs/restricted-image-only-native-vision-2026-08-21/blinded_images \
  | sort | sed -n '229,248p'
```

## Non-negotiable annotation boundary

- Assess only the pixels of the hash-named blinded image currently displayed.
- Use direct native visual inspection only; do not use local VLMs, repository staging workflows, cell counters, scripts that infer morphology, prior predictions, or external models.
- Do not open source filenames, directories, animal/day identity, acquisition sequence, or existing stage metadata.
- Do not use neighboring images or temporal plausibility to revise an image-level call.
- Treat every stage as a provisional AI visual judgment, not expert ground truth.
- Use `ungradable` for obscured or unrepresentative fields instead of forcing a four-stage label.

Basic filesystem listing, direct image display, append-only saving, JSON validation, and hashing are allowed because they do not generate or influence the biological assessment.

## Completion protocol

1. Continue direct visual review in ascending SHA-256 order and append exactly one JSON object per image.
2. Validate 343 parseable records, 343 unique hashes, and exact agreement with the full blinded inventory.
3. Freeze `annotations.jsonl` and record its final SHA-256 before opening any source labels or identity metadata.
4. Save a final completion manifest and update this handoff and `README.md`.
5. Only after the blind file is frozen may benchmarking against source metadata begin.

Do not interpret the current class distribution: the unfinished hash-ordered subset is not a completed benchmark and its labels remain provisional.
