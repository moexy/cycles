from __future__ import annotations

import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
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


def test_prepare_blind_teacher_rejects_safe_filename_collisions(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(_png_bytes())
    second.write_bytes(_png_bytes())
    source = tmp_path / "teacher.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "model_record": {"image_path": str(image)},
                    "teacher_label": {"primary_stage": "estrus"},
                }
            )
            + "\n"
            for sample_id, image in (("a b", first), ("a_b", second))
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"

    with pytest.raises(ValueError, match="safe filename collision"):
        prepare_sft_dataset("blind-teacher", source, output)

    assert not output.exists()


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


def test_benchmark_rejects_duplicate_prediction_ids(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "report"
    row = {
        "sample_id": "duplicate",
        "image_prediction": {
            "primary_stage": "estrus",
            "probabilities": {
                "diestrus": 0.1,
                "proestrus": 0.1,
                "estrus": 0.7,
                "metestrus": 0.1,
            },
        },
    }
    predictions.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    labels.write_text("sample_id,stage\nduplicate,estrus\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate prediction sample_id"):
        benchmark_predictions(predictions, labels, output)


def test_benchmark_rejects_duplicate_label_ids(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    labels = tmp_path / "labels.csv"
    output = tmp_path / "report"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": "duplicate",
                "image_prediction": {
                    "primary_stage": "estrus",
                    "probabilities": {
                        "diestrus": 0.1,
                        "proestrus": 0.1,
                        "estrus": 0.7,
                        "metestrus": 0.1,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels.write_text(
        "sample_id,stage\nduplicate,estrus\nduplicate,estrus\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate label sample_id"):
        benchmark_predictions(predictions, labels, output)


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({"diestrus": 0.1, "proestrus": 0.1, "estrus": float("nan"), "metestrus": 0.7}, "finite"),
        ({"diestrus": 1.0, "proestrus": 1.0, "estrus": 1.0, "metestrus": 7.0}, "sum to 1"),
    ],
)
def test_benchmark_rejects_invalid_probability_distributions(
    tmp_path: Path, probabilities: dict[str, float], message: str
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_prediction": {
                    "primary_stage": "metestrus",
                    "probabilities": probabilities,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,stage\ns1,metestrus\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        benchmark_predictions(predictions, labels, tmp_path / "report")


def test_benchmark_rejects_primary_stage_below_probability_mode(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_prediction": {
                    "primary_stage": "diestrus",
                    "probabilities": {
                        "diestrus": 0.1,
                        "proestrus": 0.1,
                        "estrus": 0.1,
                        "metestrus": 0.7,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,stage\ns1,metestrus\n", encoding="utf-8")

    with pytest.raises(ValueError, match="largest probability"):
        benchmark_predictions(predictions, labels, tmp_path / "report")


def test_benchmark_rejects_partial_sample_coverage(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "image_prediction": {
                    "primary_stage": "estrus",
                    "probabilities": {
                        "diestrus": 0.1,
                        "proestrus": 0.1,
                        "estrus": 0.7,
                        "metestrus": 0.1,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,stage\ns1,estrus\ns2,diestrus\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sample_id coverage"):
        benchmark_predictions(predictions, labels, tmp_path / "report")


def _provenanced_predictions(
    path: Path,
    rows: list[tuple[str, str, dict[str, float]]],
    provenance: list[dict[str, str]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample,
                    "image_prediction": {"primary_stage": stage, "probabilities": probabilities},
                    "provenance": row_provenance,
                }
            )
            + "\n"
            for (sample, stage, probabilities), row_provenance in zip(rows, provenance, strict=True)
        ),
        encoding="utf-8",
    )


def _labels_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sample_id", "stage", "group_id", "species", "stain", "lab"]
        )
        writer.writeheader()
        for sample, stage in rows:
            writer.writerow(
                {
                    "sample_id": sample,
                    "stage": stage,
                    "group_id": sample,
                    "species": "mouse",
                    "stain": "alcian blue",
                    "lab": "local",
                }
            )


_DIESTRUS = {"diestrus": 0.9, "proestrus": 0.05, "estrus": 0.03, "metestrus": 0.02}
_ESTRUS = {"diestrus": 0.05, "proestrus": 0.03, "estrus": 0.9, "metestrus": 0.02}
_ROWS = [("s1", "diestrus", _DIESTRUS), ("s2", "estrus", _ESTRUS)]


def _provenance(**overrides: str) -> dict[str, str]:
    base = {
        "model_id": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        "model_revision": "d" * 40,
        "adapter_hash": "none",
        "calibrator_hash": "none",
        "prompt_version": "morphology-first-v3",
        "prompt_prefix_reuse": "on",
        "view_pack_version": "overview-quadrants-v1",
        "schema_version": "3.0",
    }
    base.update(overrides)
    return base


def test_benchmark_rejects_a_file_that_mixes_prefill_modes(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _provenanced_predictions(
        predictions,
        _ROWS,
        [_provenance(), _provenance(prompt_prefix_reuse="off")],
    )
    labels = tmp_path / "labels.csv"
    _labels_csv(labels, [("s1", "diestrus"), ("s2", "estrus")])

    with pytest.raises(ValueError, match="mixes prompt_prefix_reuse"):
        benchmark_predictions(predictions, labels, tmp_path / "report")


def test_benchmark_rejects_a_comparison_across_prefill_modes(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _provenanced_predictions(predictions, _ROWS, [_provenance(), _provenance()])
    _provenanced_predictions(
        baseline,
        _ROWS,
        [_provenance(prompt_prefix_reuse="off"), _provenance(prompt_prefix_reuse="off")],
    )
    labels = tmp_path / "labels.csv"
    _labels_csv(labels, [("s1", "diestrus"), ("s2", "estrus")])

    with pytest.raises(ValueError, match="same frozen configuration"):
        benchmark_predictions(
            predictions, labels, tmp_path / "report", baseline_predictions=baseline
        )


def test_benchmark_compares_two_models_under_one_frozen_configuration(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _provenanced_predictions(predictions, _ROWS, [_provenance(), _provenance()])
    other = _provenance(model_id="mlx-community/gemma-3-12b-it-4bit", model_revision="8" * 40)
    _provenanced_predictions(baseline, _ROWS, [other, other])
    labels = tmp_path / "labels.csv"
    _labels_csv(labels, [("s1", "diestrus"), ("s2", "estrus")])

    report = benchmark_predictions(
        predictions, labels, tmp_path / "report", baseline_predictions=baseline
    )

    # comparing two models is the point of a bakeoff, so model identity may differ
    assert report["run_configuration"]["model_id"].endswith("Qwen3-VL-8B-Instruct-4bit")
    assert report["baseline_run_configuration"]["model_id"].endswith("gemma-3-12b-it-4bit")
    assert report["run_configuration"]["prompt_prefix_reuse"] == "on"
    assert report["gates"]["prefill_mode_declared"] is True


def test_benchmark_can_pin_the_required_prefill_mode(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    _provenanced_predictions(predictions, _ROWS, [_provenance(), _provenance()])
    labels = tmp_path / "labels.csv"
    _labels_csv(labels, [("s1", "diestrus"), ("s2", "estrus")])

    with pytest.raises(ValueError, match="requires 'off'"):
        benchmark_predictions(
            predictions, labels, tmp_path / "report", require_prefill_mode="off"
        )

    report = benchmark_predictions(
        predictions, labels, tmp_path / "report", require_prefill_mode="on"
    )
    assert report["run_configuration"]["prompt_prefix_reuse"] == "on"


def test_benchmark_flags_predictions_that_never_declared_a_prefill_mode(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample,
                    "image_prediction": {"primary_stage": stage, "probabilities": probabilities},
                }
            )
            + "\n"
            for sample, stage, probabilities in _ROWS
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.csv"
    _labels_csv(labels, [("s1", "diestrus"), ("s2", "estrus")])

    report = benchmark_predictions(predictions, labels, tmp_path / "report")

    assert report["run_configuration"]["prompt_prefix_reuse"] == "unspecified"
    assert report["gates"]["prefill_mode_declared"] is False

    with pytest.raises(ValueError, match="requires 'on'"):
        benchmark_predictions(
            predictions, labels, tmp_path / "report", require_prefill_mode="on"
        )
