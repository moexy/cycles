#!/usr/bin/env python3
"""Persist a controlled morphology-sensitivity probe with raw model responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cycles.vlm_local.calibration import TemperatureCalibrator
from cycles.vlm_local.pipeline import _parse_json_object, calibrate_stage_evidence
from cycles.vlm_local.probes import PROBE_DISCLAIMER, morphology_sensitivity_cases
from cycles.vlm_local.prompts import PROMPT_VERSION, stage_prompt
from cycles.vlm_local.schema import StageEvidence
from cycles.vlm_local.views import VIEW_PACK_VERSION, build_view_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quadrant-max-edge", type=int)
    parser.add_argument("--max-tokens", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    from cycles.vlm_local.backend import MLXVLMBackend

    backend = MLXVLMBackend(
        args.model,
        model_revision=args.model_revision,
        max_tokens=args.max_tokens,
    )
    views = build_view_pack(image_path, quadrant_max_edge=args.quadrant_max_edge)
    images = [view.image for view in views]
    calibrator = TemperatureCalibrator()
    results: list[dict[str, Any]] = []
    for case in morphology_sensitivity_cases():
        prompt = stage_prompt(case.morphology)
        started = time.perf_counter()
        raw = backend.generate(images, prompt)
        elapsed = time.perf_counter() - started
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "expected_stage": case.expected_stage.value,
            "expectation_status": "design_expectation_not_ground_truth",
            "morphology": asdict(case.morphology),
            "prompt": prompt,
            "raw_response": raw,
            "elapsed_seconds": round(elapsed, 6),
        }
        try:
            evidence = StageEvidence.from_dict(_parse_json_object(raw))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            row.update(parse_success=False, parse_error=f"{type(exc).__name__}: {exc}")
        else:
            prediction = calibrate_stage_evidence(evidence, calibrator)
            predicted = prediction.primary_stage
            row.update(
                parse_success=True,
                parsed_stage_evidence=asdict(evidence),
                derived_prediction=asdict(prediction),
                expected_match=predicted is case.expected_stage,
            )
        results.append(row)

    matches = sum(bool(row.get("expected_match")) for row in results)
    artifact = {
        "schema_version": "1.0",
        "probe_type": "counterfactual_morphology_sensitivity",
        "disclaimer": PROBE_DISCLAIMER,
        "source_image": str(image_path),
        "source_image_sha256": _sha256(image_path),
        "source_image_role": "fixed visual context; not probe ground truth",
        "model": backend.provenance,
        "prompt_version": PROMPT_VERSION,
        "score_interpretation": (
            "Model-authored relative evidence scores, not internal logits; identity-temperature "
            "softmax is an engineering default, not a validated calibration result."
        ),
        "view_pack_version": VIEW_PACK_VERSION,
        "quadrant_max_edge": args.quadrant_max_edge,
        "summary": {
            "cases": len(results),
            "parsed": sum(bool(row.get("parse_success")) for row in results),
            "design_expectation_matches": matches,
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
