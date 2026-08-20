#!/usr/bin/env python3
"""Run one non-held-out two-pass inference and persist observed MLX resource use."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    import mlx.core as mx

    from cycles.vlm_local.backend import MLXVLMBackend
    from cycles.vlm_local.pipeline import LocalVLMPipeline

    mx.reset_peak_memory()
    load_started = time.perf_counter()
    backend = MLXVLMBackend(args.model, model_revision=args.model_revision)
    load_seconds = time.perf_counter() - load_started
    peak_after_load = int(mx.get_peak_memory())

    generation_calls: list[dict[str, Any]] = []
    original_generate = backend._generate

    def instrumented_generate(*call_args: Any, **call_kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_generate(*call_args, **call_kwargs)
        generation_calls.append(
            {
                "wall_seconds": round(time.perf_counter() - started, 6),
                "prompt_tokens": _number(result, "prompt_tokens"),
                "generation_tokens": _number(result, "generation_tokens"),
                "prompt_tokens_per_second": _number(result, "prompt_tps"),
                "generation_tokens_per_second": _number(result, "generation_tps"),
            }
        )
        return result

    backend._generate = instrumented_generate

    pipeline = LocalVLMPipeline(backend, software_lock_hash=_software_lock_hash())
    inference_started = time.perf_counter()
    record = pipeline.classify_image(image_path)
    inference_seconds = time.perf_counter() - inference_started
    peak_overall = int(mx.get_peak_memory())
    gib = 1024**3

    artifact = {
        "schema_version": "1.0",
        "measurement_status": "observed_single_run_not_extrapolated",
        "source_image": str(image_path),
        "source_image_sha256": _sha256(image_path),
        "source_image_role": "non-held-out training image",
        "model": backend.provenance,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": version("mlx"),
            "mlx_vlm": version("mlx-vlm"),
        },
        "resource_usage": {
            "model_load_seconds": round(load_seconds, 6),
            "inference_seconds": round(inference_seconds, 6),
            "peak_after_load_bytes": peak_after_load,
            "peak_after_load_gib": round(peak_after_load / gib, 6),
            "peak_overall_bytes": peak_overall,
            "peak_overall_gib": round(peak_overall / gib, 6),
            "generation_call_count": len(generation_calls),
            "generation_calls": generation_calls,
        },
        "record": record.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact["resource_usage"], sort_keys=True))
    return 0


def _software_lock_hash() -> str:
    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.is_file() else "unlocked"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, attribute: str) -> int | float | None:
    item = getattr(value, attribute, None)
    if isinstance(item, (int, float)):
        return round(item, 6) if isinstance(item, float) else item
    return None


if __name__ == "__main__":
    raise SystemExit(main())
