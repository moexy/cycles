"""Path-blind content inventories for restricted annotation corpora."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from cycles.vlm_local.datasets import IMAGE_SUFFIXES


def build_blind_inventory(root: Path | str) -> dict[str, Any]:
    """Hash the image-content multiset without retaining revealing filenames."""
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    paths = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"no supported images found in {source}")
    rows = sorted(
        ({"bytes": path.stat().st_size, "image_sha256": _sha256(path)} for path in paths),
        key=lambda row: (row["image_sha256"], row["bytes"]),
    )
    encoded_rows = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    counts = Counter(row["image_sha256"] for row in rows)
    return {
        "schema_version": "1.0",
        "scope": "caller-supplied directory; paths intentionally omitted",
        "image_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "duplicate_content_groups": sum(count > 1 for count in counts.values()),
        "content_inventory_sha256": hashlib.sha256(encoded_rows).hexdigest(),
        "images": rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
