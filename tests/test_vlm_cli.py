from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

import cycles.cli.main as cli


def test_parser_recognizes_local_vlm_commands(tmp_path: Path) -> None:
    image = tmp_path / "slide.png"
    Image.new("RGB", (8, 8)).save(image)
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,stage\ns1,estrus\n")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n")
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text("{}\n")

    parsed = [
        cli.build_parser().parse_args(["vlm-local", "--input", str(image), "--model", "test/model", "--output", str(tmp_path / "out.jsonl")]),
        cli.build_parser().parse_args(["vlm-prepare-sft", "--source", "blind-teacher", "--input", str(predictions), "--output", str(tmp_path / "sft")]),
        cli.build_parser().parse_args(["vlm-benchmark", "--predictions", str(predictions), "--baseline-predictions", str(baseline), "--labels", str(labels), "--output", str(tmp_path / "report")]),
    ]

    assert [item.command for item in parsed] == ["vlm-local", "vlm-prepare-sft", "vlm-benchmark"]
    assert parsed[-1].baseline_predictions == baseline


def test_vlm_local_writes_jsonl_and_applies_sequence_manifest(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "slide.png"
    Image.new("RGB", (8, 8)).save(image)
    manifest = tmp_path / "sequence.csv"
    manifest.write_text(
        f"sample_id,image_path,subject_id,day\ns1,{image},mouse-1,4\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.jsonl"
    calibrator = tmp_path / "calibrator.json"
    calibrator.write_text('{"schema_version":"1.0","temperature":1.5}')
    record = SimpleNamespace(to_dict=lambda: {"sample_id": "s1", "sequence_prediction": {"final_stage": "estrus"}})
    pipeline = MagicMock()
    pipeline.classify_image.return_value = record
    builder = MagicMock(return_value=pipeline)
    monkeypatch.setattr(cli, "_build_local_vlm_pipeline", builder)
    reconciler = MagicMock()
    reconciler.reconcile.return_value = [record]
    monkeypatch.setattr(cli, "_build_temporal_reconciler", MagicMock(return_value=reconciler))

    status = cli.main(["vlm-local", "--input", str(image), "--model", "test/model", "--output", str(output), "--sequence-manifest", str(manifest), "--calibrator", str(calibrator)])

    assert status == 0
    builder.assert_called_once_with("test/model", None, "unspecified", calibrator)
    pipeline.classify_image.assert_called_once_with(image.resolve(), sample_id="s1", subject_id="mouse-1", day=4.0)
    reconciler.reconcile.assert_called_once_with([record])
    assert json.loads(output.read_text()) == record.to_dict()
