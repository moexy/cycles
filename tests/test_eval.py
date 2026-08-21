from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cyclonaut.eval.benchmark import BenchmarkHarness
from cyclonaut.eval.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
)

CANONICAL = ["diestrus", "proestrus", "estrus", "metestrus"]


def _save_image(path: Path, color: tuple[int, int, int] = (40, 80, 120)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)
    return path


def test_compute_classification_metrics_diverse_labels_and_scores() -> None:
    y_true = ["diestrus", "diestrus", "proestrus", "estrus", "metestrus", "metestrus"]
    y_pred = ["diestrus", "proestrus", "proestrus", "estrus", "estrus", "metestrus"]

    metrics = compute_classification_metrics(y_true, y_pred, labels=CANONICAL)

    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.balanced_accuracy == pytest.approx(3 / 4)
    assert metrics.cohens_kappa == pytest.approx(4 / 7)
    assert metrics.macro_f1 == pytest.approx(2 / 3)
    assert metrics.weighted_f1 == pytest.approx(2 / 3)
    assert metrics.macro_precision == pytest.approx(3 / 4)
    assert metrics.macro_recall == pytest.approx(3 / 4)
    assert metrics.class_labels == CANONICAL
    assert metrics.confusion_matrix == [
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
    ]


def test_compute_metrics_with_single_class_predictions() -> None:
    metrics = compute_classification_metrics(
        CANONICAL,
        ["diestrus"] * 4,
        labels=CANONICAL,
    )

    assert metrics.accuracy == pytest.approx(0.25)
    assert metrics.balanced_accuracy == pytest.approx(0.25)
    assert metrics.cohens_kappa == pytest.approx(0.0)
    assert metrics.macro_f1 == pytest.approx(0.1)
    assert metrics.weighted_f1 == pytest.approx(0.1)


def test_compute_metrics_constant_single_sample_has_finite_kappa() -> None:
    metrics = compute_classification_metrics(["estrus"], ["estrus"])

    assert metrics.accuracy == 1.0
    assert metrics.balanced_accuracy == 1.0
    assert metrics.cohens_kappa == 0.0
    assert metrics.macro_f1 == 1.0
    assert metrics.confusion_matrix == [[1]]


def test_compute_metrics_empty_arrays_with_and_without_labels() -> None:
    unlabeled = compute_classification_metrics([], [])
    labeled = compute_classification_metrics([], [], labels=CANONICAL)

    assert unlabeled.class_labels == []
    assert unlabeled.confusion_matrix == []
    assert unlabeled.accuracy == unlabeled.throughput_fps == 0.0
    assert labeled.class_labels == CANONICAL
    assert labeled.confusion_matrix == [[0, 0, 0, 0] for _ in CANONICAL]
    assert labeled.macro_f1 == labeled.weighted_f1 == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"y_true": ["estrus"], "y_pred": []}, "same length"),
        (
            {"y_true": ["estrus"], "y_pred": ["estrus"], "confidences": []},
            "confidences must contain one value",
        ),
        (
            {"y_true": ["estrus"], "y_pred": ["estrus"], "latencies": []},
            "latencies must contain one value",
        ),
    ],
)
def test_compute_metrics_rejects_mismatched_parallel_arrays(
    kwargs: dict[str, list[str] | list[float]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_classification_metrics(**kwargs)  # type: ignore[arg-type]


def test_compute_metrics_validates_explicit_labels() -> None:
    with pytest.raises(ValueError, match="does not include observed classes"):
        compute_classification_metrics(["estrus"], ["estrus"], labels=["diestrus"])


def test_compute_metrics_filters_nonfinite_confidences_and_latencies() -> None:
    metrics = compute_classification_metrics(
        ["diestrus", "proestrus", "estrus", "metestrus"],
        ["diestrus", "proestrus", "estrus", "metestrus"],
        confidences=[0.8, float("nan"), 0.4, float("inf")],
        latencies=[0.01, float("nan"), 0.02, -1.0],
        labels=CANONICAL,
    )

    assert metrics.mean_confidence == pytest.approx(0.6)
    assert metrics.latency_ms_per_slide == pytest.approx(15.0)
    assert metrics.throughput_fps == pytest.approx(1 / 0.015)


def test_plot_confusion_matrix_saves_valid_png(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "confusion.png"

    returned = plot_confusion_matrix([[3, 1], [2, 4]], ["diestrus", "estrus"], output)

    assert returned == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(output) as rendered:
        assert rendered.format == "PNG"
        assert rendered.width > 100
        assert rendered.height > 100


def test_plot_confusion_matrix_rejects_wrong_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        plot_confusion_matrix([[1, 0]], ["diestrus", "estrus"], tmp_path / "bad.png")


def test_plot_model_comparison_saves_valid_png(tmp_path: Path) -> None:
    metrics_by_model = {
        "accurate": compute_classification_metrics(CANONICAL, CANONICAL, labels=CANONICAL),
        "constant": compute_classification_metrics(CANONICAL, ["diestrus"] * 4, labels=CANONICAL),
    }
    output = tmp_path / "comparison.png"

    returned = plot_model_comparison(metrics_by_model, output)

    assert returned == output
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_model_comparison_requires_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one model"):
        plot_model_comparison({}, tmp_path / "empty.png")


class _FilenameModel:
    def __init__(self, *, constant_stage: str | None = None) -> None:
        self.constant_stage = constant_stage

    def process_image(self, image_path: Path) -> SimpleNamespace:
        if "corrupt" in image_path.name:
            raise OSError("cannot decode synthetic corrupt image")
        stage = self.constant_stage or image_path.parent.name
        probabilities = {label: (0.91 if label == stage else 0.03) for label in CANONICAL}
        return SimpleNamespace(predicted_stage=stage, confidence=0.91, probabilities=probabilities)


def test_benchmark_harness_generates_reports_and_isolates_corrupt_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "dataset"
    _save_image(image_dir / "diestrus" / "d_01.png")
    _save_image(image_dir / "proestrus" / "p_01.png")
    _save_image(image_dir / "estrus" / "e_01.png")
    corrupt = image_dir / "metestrus" / "m_corrupt.png"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not an image")
    output_dir = tmp_path / "reports"
    monkeypatch.setattr(
        BenchmarkHarness,
        "_hardware_info",
        lambda self: {"platform": "test", "accelerator": "CPU"},
    )
    harness = BenchmarkHarness(
        image_dir=image_dir,
        output_dir=output_dir,
        models=["resnet50", "mil"],
        model_instances={
            "resnet50": _FilenameModel(),
            "mil": _FilenameModel(constant_stage="diestrus"),
        },
    )

    report = harness.run()

    assert report.total_slides == 4
    assert report.labeled_slides == 4
    assert report.metrics_by_model["resnet50"].accuracy == 1.0
    assert report.metrics_by_model["mil"].accuracy == pytest.approx(1 / 3)
    assert report.failures["resnet50"]["metestrus/m_corrupt.png"].startswith("OSError:")
    assert report.failures["mil"]["metestrus/m_corrupt.png"].startswith("OSError:")
    assert report.csv_path and report.csv_path.is_file()
    assert report.json_path and report.json_path.is_file()
    assert report.markdown_path and report.markdown_path.is_file()
    assert len(report.plot_paths) == 3
    assert all(path.is_file() and path.read_bytes().startswith(b"\x89PNG") for path in report.plot_paths)

    with report.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["ground_truth"] for row in rows} == set(CANONICAL)
    assert next(row for row in rows if row["image"] == "m_corrupt.png")["mil_error"].startswith(
        "OSError:"
    )

    payload = json.loads(report.json_path.read_text(encoding="utf-8"))
    assert payload["total_slides"] == 4
    assert payload["models"]["resnet50"]["accuracy"] == 1.0
    assert len(payload["predictions"]) == 4
    markdown = report.markdown_path.read_text(encoding="utf-8")
    assert "# Estrous Stage Benchmark Report" in markdown
    assert "| resnet50 | 1.000" in markdown
    assert "## Failures" in markdown


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        ("diestrus/slide_01.png", "diestrus"),
        ("nested/PROESTRUS/sample.png", "proestrus"),
        ("unlabeled/estrus_mouse7.png", "estrus"),
        ("unlabeled/m_004.tif", "metestrus"),
        ("unlabeled/d-004.ome.tiff", "diestrus"),
        ("unlabeled/protocol.png", None),
    ],
)
def test_ground_truth_from_folder_and_filename_prefixes(
    tmp_path: Path, relative_path: str, expected: str | None
) -> None:
    root = tmp_path / "dataset"
    path = root / relative_path

    assert BenchmarkHarness.parse_ground_truth(path, root) == expected


def test_annotation_csv_ground_truth_supports_paths_names_and_aliases(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    annotations.write_text(
        "image_path,ground_truth\n"
        "nested/slide-a.png,proestrus\n"
        "slide-b.png,D\n"
        "slide-c.png,met\n",
        encoding="utf-8",
    )

    loaded = BenchmarkHarness.load_annotations(annotations)

    assert loaded["nested/slide-a.png"] == "proestrus"
    assert loaded["slide-a.png"] == "proestrus"
    assert loaded["slide-b.png"] == "diestrus"
    assert loaded["slide-c.png"] == "metestrus"


def test_annotation_csv_rejects_missing_columns_and_invalid_stage(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("subject,value\nmouse,estrus\n", encoding="utf-8")
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("filename,stage\nslide.png,unknown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires an image column"):
        BenchmarkHarness.load_annotations(missing)
    with pytest.raises(ValueError, match="invalid estrous stage"):
        BenchmarkHarness.load_annotations(invalid)
