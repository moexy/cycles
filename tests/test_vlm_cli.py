from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

import cycles.cli.main as cli

REVISION = "a" * 40


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
        cli.build_parser().parse_args(["vlm-local", "--input", str(image), "--model", "test/model", "--model-revision", REVISION, "--output", str(tmp_path / "out.jsonl")]),
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

    status = cli.main(["vlm-local", "--input", str(image), "--model", "test/model", "--model-revision", REVISION, "--output", str(output), "--sequence-manifest", str(manifest), "--calibrator", str(calibrator)])

    assert status == 0
    builder.assert_called_once_with(
        "test/model", None, REVISION, calibrator, reuse_prompt_prefix=True
    )
    pipeline.classify_image.assert_called_once_with(image.resolve(), sample_id="s1", subject_id="mouse-1", day=4.0)
    reconciler.reconcile.assert_called_once_with([record])
    assert json.loads(output.read_text()) == record.to_dict()


def test_vlm_local_can_disable_prompt_prefix_reuse(monkeypatch, tmp_path: Path) -> None:
    """A frozen run must be able to force cold prefill to match older records."""
    image = tmp_path / "slide.png"
    Image.new("RGB", (8, 8)).save(image)
    output = tmp_path / "results.jsonl"
    record = SimpleNamespace(
        to_dict=lambda: {"sample_id": "s1", "sequence_prediction": {"final_stage": "estrus"}}
    )
    pipeline = MagicMock()
    pipeline.classify_image.return_value = record
    builder = MagicMock(return_value=pipeline)
    monkeypatch.setattr(cli, "_build_local_vlm_pipeline", builder)

    status = cli.main(
        [
            "vlm-local",
            "--input", str(image),
            "--model", "test/model",
            "--model-revision", REVISION,
            "--output", str(output),
            "--no-prompt-prefix-reuse",
        ]
    )

    assert status == 0
    assert builder.call_args.kwargs["reuse_prompt_prefix"] is False


def test_vlm_local_requires_model_revision(tmp_path: Path) -> None:
    image = tmp_path / "slide.png"
    Image.new("RGB", (8, 8)).save(image)

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "vlm-local",
                "--input", str(image),
                "--model", "test/model",
                "--output", str(tmp_path / "out.jsonl"),
            ]
        )


def test_software_lock_hash_does_not_depend_on_working_directory(
    monkeypatch, tmp_path: Path
) -> None:
    expected = cli._software_lock_hash()
    monkeypatch.chdir(tmp_path)

    assert cli._software_lock_hash() == expected
    assert expected != "unlocked"


def test_vlm_calibrate_cli(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "calibrator.json"

    rows = [
        {"sample_id": "s1", "image_prediction": {"raw_scores": {"diestrus": 8.0, "proestrus": 0.0, "estrus": 0.0, "metestrus": 0.0}}},
        {"sample_id": "s2", "image_prediction": {"raw_scores": {"diestrus": 0.0, "proestrus": 8.0, "estrus": 0.0, "metestrus": 0.0}}},
    ]
    predictions.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    labels.write_text("sample_id,stage\ns1,diestrus\ns2,proestrus\n", encoding="utf-8")

    status = cli.main([
        "vlm-calibrate",
        "--predictions", str(predictions),
        "--labels", str(labels),
        "--output", str(output),
    ])

    assert status == 0
    assert output.is_file()
    data = json.loads(output.read_text())
    assert data["schema_version"] == "1.0"
    assert data["temperature"] > 0


def test_vlm_local_resume_skips_existing_records(tmp_path: Path, monkeypatch) -> None:
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    Image.new("RGB", (8, 8)).save(img1)
    Image.new("RGB", (8, 8)).save(img2)

    folder = tmp_path / "input_images"
    folder.mkdir()
    (folder / "img1.png").write_bytes(img1.read_bytes())
    (folder / "img2.png").write_bytes(img2.read_bytes())

    output = tmp_path / "resumed_output.jsonl"

    rec1 = {
        "schema_version": "3.0",
        "sample_id": "img1",
        "image_path": str((folder / "img1.png").resolve()),
        "image_sha256": "0" * 64,
        "subject_id": None,
        "day": None,
        "morphology": {
            "cornified_squames": "absent",
            "nucleated_epithelial": "absent",
            "leukocytes": "absent",
            "nuclear_state": "not_assessable",
            "arrangement": "not_assessable",
            "artifacts": [],
            "qc_status": "usable",
            "qc_reasons": [],
            "evidence": [],
        },
        "image_prediction": {
            "primary_stage": "estrus",
            "secondary_stage": None,
            "raw_scores": {"diestrus": 0.0, "proestrus": 0.0, "estrus": 2.0, "metestrus": 0.0},
            "probabilities": {"diestrus": 0.1, "proestrus": 0.1, "estrus": 0.7, "metestrus": 0.1},
            "confidence_tier": "high",
            "rationale": "Cornified squames only",
        },
        "sequence_prediction": {
            "final_stage": "estrus",
            "adjusted": False,
            "reason": "image_only",
        },
        "provenance": {"prompt_version": "test", "schema_version": "3.0"},
    }
    output.write_text(json.dumps(rec1) + "\n", encoding="utf-8")

    from cycles.vlm_local.schema import LocalVLMRecord
    rec2 = LocalVLMRecord.from_dict({**rec1, "sample_id": "img2", "image_path": str((folder / "img2.png").resolve())})

    pipeline = MagicMock()
    pipeline.classify_image.return_value = rec2
    builder = MagicMock(return_value=pipeline)
    monkeypatch.setattr(cli, "_build_local_vlm_pipeline", builder)

    status = cli.main([
        "vlm-local",
        "--input", str(folder),
        "--model", "test/model",
        "--model-revision", REVISION,
        "--output", str(output),
        "--resume",
    ])

    assert status == 0
    # Should only classify img2, since img1 was already present in output
    assert pipeline.classify_image.call_count == 1
    call_args = pipeline.classify_image.call_args[0]
    assert call_args[0] == (folder / "img2.png").resolve()

    lines = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert len(lines) == 2
    assert {row["sample_id"] for row in lines} == {"img1", "img2"}

