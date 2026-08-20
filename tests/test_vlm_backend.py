from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from PIL import Image

from cycles.vlm_local.backend import MLXVLMBackend


def test_mlx_backend_materializes_lossless_paths_for_installed_api(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def load(model_id: str, **kwargs):
        captured["load"] = (model_id, kwargs)
        return object(), object()

    def generate(model, processor, prompt: str, *, image, **kwargs):
        paths = [Path(value) for value in image]
        assert all(path.is_file() and path.suffix == ".png" for path in paths)
        assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
        captured["paths"] = paths
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return SimpleNamespace(text='{"ok": true}')

    fake = ModuleType("mlx_vlm")
    fake.load = load  # type: ignore[attr-defined]
    fake.generate = generate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake)
    backend = MLXVLMBackend("test/model", model_revision="rev-1")

    response = backend.generate(
        [Image.new("RGB", (8, 8)), Image.new("RGB", (4, 4))],
        "inspect",
    )

    assert response == '{"ok": true}'
    assert captured["load"] == ("test/model", {"revision": "rev-1"})
    assert captured["prompt"] == "inspect"
    assert captured["kwargs"] == {"max_tokens": 1024, "verbose": False}
    assert all(not path.exists() for path in captured["paths"])
