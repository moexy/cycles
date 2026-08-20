"""Backend boundary for optional MLX-VLM inference."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

from PIL import Image


class VLMBackend(Protocol):
    @property
    def provenance(self) -> dict[str, str]: ...

    def generate(self, images: Sequence[Image.Image], prompt: str) -> str: ...


class MLXVLMBackend:
    """Lazy MLX-VLM adapter; importing cycles does not require MLX."""

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: Path | str | None = None,
        model_revision: str = "unspecified",
        max_tokens: int = 1024,
        reuse_prompt_prefix: bool = True,
    ) -> None:
        try:
            from mlx_vlm import apply_chat_template, generate, load
            from mlx_vlm.generate.common import PromptCacheState
        except ImportError as exc:
            raise RuntimeError(
                "Local VLM inference requires the optional 'mlx' dependencies on Apple Silicon"
            ) from exc
        self._generate = generate
        self._apply_chat_template = apply_chat_template
        self._model_id = model_id
        self._model_revision = model_revision
        self._adapter_path = Path(adapter_path).expanduser() if adapter_path else None
        load_kwargs: dict[str, Any] = {}
        if self._adapter_path is not None:
            load_kwargs["adapter_path"] = str(self._adapter_path)
        if model_revision != "unspecified":
            load_kwargs["revision"] = model_revision
        self._model, self._processor = load(model_id, **load_kwargs)
        self._max_tokens = max_tokens
        self._prompt_cache_state_factory = PromptCacheState
        self._reuse_prompt_prefix = reuse_prompt_prefix
        self._cache_signature: tuple[str, ...] | None = None
        self._prompt_cache_state: Any = None

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "model_id": self._model_id,
            "model_revision": self._model_revision,
            "adapter_hash": _path_hash(self._adapter_path) if self._adapter_path else "none",
            # Reusing the KV prefix is a different, self-consistent numeric path:
            # repeated runs agree, but reused and cold prefill can disagree on
            # knife-edge images. Records are only comparable at a fixed setting,
            # so the setting travels with the record.
            "prompt_prefix_reuse": "on" if self._reuse_prompt_prefix else "off",
        }

    def generate(self, images: Sequence[Image.Image], prompt: str) -> str:
        # mlx-vlm 0.6.15 accepts paths/URLs rather than PIL objects. Lossless
        # temporary PNGs preserve the deterministic view pack without JPEG
        # recompression and are removed immediately after generation.
        with TemporaryDirectory(prefix="cycles-mlx-views-") as temporary:
            paths = []
            digests = []
            for index, image in enumerate(images):
                path = Path(temporary) / f"view-{index}.png"
                image.save(path, format="PNG")
                paths.append(str(path))
                digests.append(hashlib.sha256(image.tobytes()).hexdigest())
            signature = tuple(digests)
            # generate() does not apply the chat template; without it the prompt
            # carries no image placeholder tokens and the vision embedding has
            # nowhere to scatter, so the model raises a broadcast error.
            formatted = self._apply_chat_template(
                self._processor,
                self._model.config,
                prompt,
                num_images=len(paths),
            )
            result = self._generate(
                self._model,
                self._processor,
                formatted,
                image=paths,
                max_tokens=self._max_tokens,
                verbose=False,
                prompt_cache_state=self._cache_state_for(signature),
            )
            return str(getattr(result, "text", result))

    def _cache_state_for(self, signature: tuple[str, ...]) -> Any:
        """Reuse the KV prefix while consecutive calls share the same views.

        Both passes over one slide send an identical view pack and differ only in
        the trailing instruction, so the image tokens are a common prefix worth
        thousands of tokens of prefill. mlx-vlm reuses it only when the diverging
        suffix is text-only, which holds here. A new slide changes the signature
        and gets a fresh cache, so no state crosses images.
        """
        if not self._reuse_prompt_prefix:
            return None
        if signature != self._cache_signature:
            self._cache_signature = signature
            self._prompt_cache_state = self._prompt_cache_state_factory()
        return self._prompt_cache_state


def _path_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            digest.update(child.read_bytes())
    else:
        raise FileNotFoundError(path)
    return digest.hexdigest()
