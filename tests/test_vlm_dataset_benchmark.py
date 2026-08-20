from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from cycles.vlm_local.benchmark import benchmark_predictions
from cycles.vlm_local.datasets import prepare_sft_dataset


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (12, 10), (20, 60, 100)).save(stream, format="PNG")
    return stream.getvalue()


def test_prepare_estrousbank_discards_templated_rationale(tmp_path: Path) -> None:
    source = tmp_path / "estrousbank"
    source.mkdir()
    shard = source / "estrousbank-train-000000.tar"
    metadata = json.dumps(
        {
            "stage": "estrus",
            "caption": "Template generated directly from the stage",
            "reasoning": "Also templated",
        }
    ).encode()
    with tarfile.open(shard, "w") as archive:
        for name, data in (("sample-1.png", _png_bytes()), ("sample-1.json", metadata)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    summary = prepare_sft_dataset("estrousbank", source, tmp_path / "prepared")

    row = json.loads((tmp_path / "prepared" / "train.jsonl").read_text())
    assistant = row["messages"][1]["content"]
    assert summary["samples_by_split"] == {"train": 1}
    assert json.loads(assistant) == {"primary_stage": "estrus"}
    assert "Template" not in assistant and "reasoning" not in assistant
    assert (tmp_path / "prepared" / row["images"][0]).read_bytes().startswith(b"\x89PNG")


def test_prepare_blind_teacher_emits_morphology_supervision(tmp_path: Path) -> None:
    image = tmp_path / "slide.png"
    image.write_bytes(_png_bytes())
    source = tmp_path / "teacher.jsonl"
    source.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "model_record": {"image_path": str(image)},
                "teacher_label": {
                    "primary_stage": "metestrus",
                    "secondary_stage": "diestrus",
                    "confidence_tier": "medium",
                    "leukocytes": "present",
                    "cornified_squames": "present",
                    "qc_status": "usable",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = prepare_sft_dataset("blind-teacher", source, tmp_path / "prepared")

    row = json.loads((tmp_path / "prepared" / "train.jsonl").read_text())
    answer = json.loads(row["messages"][1]["content"])
    assert summary["samples_by_split"] == {"train": 1}
    assert answer["primary_stage"] == "metestrus"
    assert answer["leukocytes"] == "present"
    assert row["metadata"]["supervision"] == "teacher"


def test_benchmark_reports_calibration_and_subgroups(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        ("s1", "diestrus", {"diestrus": 0.9, "proestrus": 0.05, "estrus": 0.03, "metestrus": 0.02}),
        ("s2", "estrus", {"diestrus": 0.05, "proestrus": 0.03, "estrus": 0.9, "metestrus": 0.02}),
    ]
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample,
                    "image_prediction": {
                        "primary_stage": stage,
                        "probabilities": probabilities,
                    },
                }
            )
            + "\n"
            for sample, stage, probabilities in rows
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "stage", "group_id", "species", "stain", "lab"])
        writer.writeheader()
        writer.writerow({"sample_id": "s1", "stage": "diestrus", "group_id": "g1", "species": "mouse", "stain": "alcian blue", "lab": "local"})
        writer.writerow({"sample_id": "s2", "stage": "estrus", "group_id": "g2", "species": "mouse", "stain": "alcian blue", "lab": "local"})

    report = benchmark_predictions(predictions, labels, tmp_path / "report")

    assert report["metrics"]["macro_f1"] == 0.5
    assert report["calibration"]["ece"] == 0.1
    assert report["calibration"]["brier_score"] > 0
    assert report["subgroups"]["species"]["mouse"]["count"] == 2
    assert (tmp_path / "report" / "report.json").is_file()


def test_benchmark_uses_group_bootstrap_and_subgroup_safety_gates(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    stages = ["diestrus", "proestrus", "estrus", "metestrus"]
    with labels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "stage", "group_id", "species", "stain", "lab"],
        )
        writer.writeheader()
        for index, stage in enumerate(stages):
            writer.writerow(
                {
                    "sample_id": f"s{index}",
                    "stage": stage,
                    "group_id": f"g{index}",
                    "species": "mouse",
                    "stain": "alcian blue",
                    "lab": "local",
                }
            )

    def write_predictions(path: Path, predicted: list[str]) -> None:
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": f"s{index}",
                        "image_prediction": {
                            "primary_stage": stage,
                            "probabilities": {
                                candidate: 0.9 if candidate == stage else 0.1 / 3
                                for candidate in stages
                            },
                        },
                    }
                )
                + "\n"
                for index, stage in enumerate(predicted)
            )
        )

    candidate = tmp_path / "candidate.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    write_predictions(candidate, stages)
    write_predictions(baseline, stages[1:] + stages[:1])

    report = benchmark_predictions(
        candidate,
        labels,
        tmp_path / "report",
        baseline_predictions=baseline,
        bootstrap_samples=100,
        min_subgroup_size=1,
    )

    assert report["comparison"]["macro_f1_delta"] == 1.0
    assert report["comparison"]["group_bootstrap_95_ci"][0] > 0
    assert report["comparison"]["group_bootstrap_95_ci"][1] <= 1.0
    assert report["gates"]["relative_improvement"] is True
    assert report["gates"]["subgroup_safety"] is True
