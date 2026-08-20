from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

from cycles.vlm_local.backend import MLXVLMBackend

IMAGE_TOKEN = "<|image_pad|>"


def _fake_mlx_vlm(captured: dict[str, object]) -> ModuleType:
    """Stand in for mlx_vlm 0.6.15, mirroring its real calling contract.

    The library's generate() does not apply the chat template. A prompt that
    reaches the model without image placeholder tokens makes get_input_embeddings
    scatter the vision embedding into an empty position set, so the fake fails
    the same way the real model does.
    """

    def load(model_id: str, **kwargs):
        captured["load"] = (model_id, kwargs)
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"))
        return model, object()

    def apply_chat_template(processor, config, prompt, *, num_images: int = 0, **kwargs):
        captured["template"] = {"config": config, "prompt": prompt, "num_images": num_images}
        return f"<|im_start|>user{IMAGE_TOKEN * num_images}{prompt}<|im_end|>"

    def generate(model, processor, prompt: str, *, image, **kwargs):
        paths = [Path(value) for value in image]
        assert all(path.is_file() and path.suffix == ".png" for path in paths)
        assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
        if prompt.count(IMAGE_TOKEN) != len(paths):
            raise ValueError(
                "[broadcast_shapes] Shapes (22097920) and (0) cannot be broadcast."
            )
        captured["paths"] = paths
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return SimpleNamespace(text='{"ok": true}')

    fake = ModuleType("mlx_vlm")
    fake.load = load  # type: ignore[attr-defined]
    fake.generate = generate  # type: ignore[attr-defined]
    fake.apply_chat_template = apply_chat_template  # type: ignore[attr-defined]
    return fake


class FakePromptCacheState:
    """Mirrors mlx_vlm.generate.common.PromptCacheState's identity semantics."""

    instances = 0

    def __init__(self) -> None:
        type(self).instances += 1
        self.cache = None
        self.token_ids = None


def _install_fake_mlx_vlm(monkeypatch, captured: dict[str, object]) -> None:
    monkeypatch.setitem(sys.modules, "mlx_vlm", _fake_mlx_vlm(captured))
    common = ModuleType("mlx_vlm.generate.common")
    common.PromptCacheState = FakePromptCacheState  # type: ignore[attr-defined]
    generate_pkg = ModuleType("mlx_vlm.generate")
    generate_pkg.common = common  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate_pkg)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.common", common)


def test_mlx_backend_materializes_lossless_paths_for_installed_api(monkeypatch) -> None:
    captured: dict[str, object] = {}
    _install_fake_mlx_vlm(monkeypatch, captured)
    backend = MLXVLMBackend("test/model", model_revision="rev-1")

    response = backend.generate(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (4, 4))],
        "inspect",
    )

    assert response == '{"ok": true}'
    assert captured["load"] == ("test/model", {"revision": "rev-1"})
    kwargs = dict(captured["kwargs"])  # type: ignore[arg-type]
    kwargs.pop("prompt_cache_state")
    assert kwargs == {"max_tokens": 1024, "verbose": False}
    assert all(not path.exists() for path in captured["paths"])


def test_mlx_backend_applies_chat_template_with_one_token_per_view(monkeypatch) -> None:
    """Each materialized view needs its own placeholder or the scatter fails."""
    captured: dict[str, object] = {}
    _install_fake_mlx_vlm(monkeypatch, captured)
    backend = MLXVLMBackend("test/model")

    backend.generate([Image.new("RGB", (8, 8))] * 4, "inspect")

    template = captured["template"]
    assert template["num_images"] == 4  # type: ignore[index]
    assert template["prompt"] == "inspect"  # type: ignore[index]
    assert str(captured["prompt"]).count(IMAGE_TOKEN) == 4


def test_prompt_cache_is_shared_across_passes_and_reset_between_slides(monkeypatch) -> None:
    """Both passes over one slide share the image prefix; a new slide must not."""
    captured: dict[str, object] = {}
    _install_fake_mlx_vlm(monkeypatch, captured)
    FakePromptCacheState.instances = 0
    backend = MLXVLMBackend("test/model")

    slide = [Image.new("RGB", (8, 8), "red")]
    first = backend._cache_state_for(("digest-a",))
    second = backend._cache_state_for(("digest-a",))
    assert first is second, "the second pass must reuse the first pass's KV prefix"

    third = backend._cache_state_for(("digest-b",))
    assert third is not first, "a different slide must not inherit cached state"
    assert FakePromptCacheState.instances == 2

    backend.generate(slide, "inspect")
    assert captured["kwargs"]["prompt_cache_state"] is not None  # type: ignore[index]
