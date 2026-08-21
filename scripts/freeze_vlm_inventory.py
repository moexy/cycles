#!/usr/bin/env python3
"""Freeze a path-blind image-content inventory for restricted annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyclonaut.vlm_local.inventory import build_blind_inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inventory = build_blind_inventory(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(inventory, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: inventory[key]
                for key in (
                    "image_count",
                    "total_bytes",
                    "duplicate_content_groups",
                    "content_inventory_sha256",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
