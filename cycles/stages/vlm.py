"""Vision-Language Model (VLM) image interpretation engine for rodent vaginal cytology."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from cycles.core.types import BatchClassificationResult, ClassificationResult, EstrousStage

LOGGER = logging.getLogger(__name__)

CYTOLOGY_SYSTEM_PROMPT = """You are an expert rodent reproductive endocrinologist and cytopathologist specializing in vaginal cytology smear staging in female rodents (mice and rats).

You classify smears into the 4 canonical estrous cycle phases based on cellular morphology and proportions:
1. DIESTRUS: Characterized predominantly by small, dense, round polymorphonuclear leukocytes (>50-80%), with occasional nucleated epithelial cells and mucus. Low estrogen.
2. PROESTRUS: Characterized predominantly by round/oval nucleated epithelial cells in sheets or clusters (high nuclear-to-cytoplasmic ratio, visible cytoplasm halo), with few cornified cells and minimal leukocytes. High estrogen.
3. ESTRUS: Characterized almost exclusively by large, flat, irregular, polygonal/angular, anucleated cornified squamous epithelial cells (>70-95%), with virtually zero leukocytes. Peak estrogen / ovulation.
4. METESTRUS: Characterized by the co-occurrence of polymorphonuclear leukocytes and cornified squamous cells in roughly equal or moderate proportions (transition phase emerging from estrus into diestrus).

Analyze the provided cytology image carefully regardless of the histological stain (H&E, Shorr, Giemsa, Crystal Violet, Cresyl Violet, Alcian Blue, or unstained phase contrast).
Respond ONLY with a valid JSON object in the following format:
{
  "predicted_stage": "diestrus" | "proestrus" | "estrus" | "metestrus",
  "confidence": float between 0.0 and 1.0,
  "probabilities": {
    "diestrus": float,
    "proestrus": float,
    "estrus": float,
    "metestrus": float
  },
  "cellular_breakdown": {
    "leukocyte_pct": float,
    "nucleated_epithelial_pct": float,
    "cornified_squamous_pct": float
  },
  "visual_rationale": "Clear expert cytological description of observed cell types, stains, boundaries, and staging decision."
}
"""


@dataclass(slots=True)
class VLMConfig:
    """Configuration for VLM multimodal cytology interpretation."""

    endpoint_url: str | None = None
    api_key: str | None = None
    model_name: str = "gpt-4o"
    timeout_seconds: float = 45.0
    max_image_dim: int = 1024


class VLMInterpretationService:
    """Zero-shot and few-shot multimodal cytology interpretation service."""

    def __init__(self, config: VLMConfig | None = None) -> None:
        self.config = config or VLMConfig()
        self.endpoint_url = self.config.endpoint_url or os.environ.get(
            "VLM_ENDPOINT_URL",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        )
        self.api_key = self.config.api_key or os.environ.get(
            "OPENAI_API_KEY",
            os.environ.get("VLM_API_KEY", ""),
        )

    def _encode_image(self, image_path: Path | str) -> str:
        """Load and resize image to JPEG base64 string for efficient VLM payload transfer."""
        path = Path(image_path).expanduser()
        with Image.open(path) as raw_img:
            img = ImageOps.exif_transpose(raw_img).convert("RGB")
            w, h = img.size
            max_dim = self.config.max_image_dim
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=88)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def interpret_image(self, image_path: Path | str) -> ClassificationResult:
        """Interpret a single vaginal cytology smear image using VLM vision inference."""
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Cytology image not found: {path}")

        # If endpoint requires auth and no key is provided, fail with a helpful descriptive error
        is_local = "localhost" in self.endpoint_url or "127.0.0.1" in self.endpoint_url
        if not self.api_key and not is_local:
            raise RuntimeError(
                "VLM vision interpretation requires an API key or local endpoint. "
                "Please set OPENAI_API_KEY, VLM_API_KEY, or configure a local endpoint "
                "(VLM_ENDPOINT_URL=http://localhost:11434/v1/chat/completions)."
            )

        base64_img = self._encode_image(path)
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": CYTOLOGY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify the rodent estrous cycle stage for this vaginal cytology smear. Provide structured probabilities and rationale.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                        },
                    ],
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }

        try:
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={k: v for k, v in headers.items() if v},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return self._build_result_from_parsed(path, parsed)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            raise RuntimeError(f"VLM API HTTP {exc.code} error from {self.endpoint_url}: {err_body}") from exc
        except Exception as exc:
            raise RuntimeError(f"VLM API request failed on {self.endpoint_url}: {exc}") from exc

    def _build_result_from_parsed(self, path: Path, parsed: dict[str, Any]) -> ClassificationResult:
        stage_str = str(parsed.get("predicted_stage", "diestrus")).strip().lower()
        try:
            stage = EstrousStage(stage_str)
        except ValueError:
            stage = EstrousStage.DIESTRUS

        conf = float(parsed.get("confidence", 0.75))
        raw_probs = parsed.get("probabilities", {})
        probs: dict[EstrousStage, float] = {}
        for s in EstrousStage.canonical_stages():
            probs[s] = float(raw_probs.get(s.value, 0.05))
        total_p = sum(probs.values())
        if total_p > 0:
            probs = {k: v / total_p for k, v in probs.items()}

        sorted_p = sorted(probs.values(), reverse=True)
        conf_idx = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0

        return ClassificationResult(
            image_path=path,
            predicted_stage=stage,
            confidence=conf,
            probabilities=probs,
            confidence_index=conf_idx,
            is_transition=conf_idx < 0.25,
            transition_to=sorted(probs, key=probs.__getitem__, reverse=True)[1] if conf_idx < 0.25 else None,
        )

    def interpret_folder(
        self,
        folder_path: Path | str,
        recursive: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> BatchClassificationResult:
        """Interpret all images in a folder."""
        folder = Path(folder_path).expanduser()
        if not folder.is_dir():
            raise NotADirectoryError(f"Folder not found: {folder}")
        from cycles.core.preprocessing import discover_images

        image_paths = discover_images(folder, recursive=recursive)
        results: list[ClassificationResult] = []
        failed: list[tuple[Path, str]] = []

        total = len(image_paths)
        import time

        t0 = time.perf_counter()
        for idx, path in enumerate(image_paths, start=1):
            if progress_callback:
                progress_callback(idx - 1, total, f"Interpreting {path.name}")
            try:
                results.append(self.interpret_image(path))
            except Exception as exc:
                failed.append((path, str(exc)))

        if progress_callback:
            progress_callback(total, total, "VLM interpretation complete")

        duration = time.perf_counter() - t0
        return BatchClassificationResult(
            results=results,
            failed_images=failed,
            total_processed=len(results) + len(failed),
            duration_seconds=duration,
        )
