from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import cycles.cli.main as cli


def test_argument_parser_recognizes_every_subcommand(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    cycle_csv = tmp_path / "cycle.csv"
    cycle_csv.write_text("day,stage\n1,diestrus\n", encoding="utf-8")
    parser = cli.build_parser()
    commands = {
        "classify": [
            "classify",
            "--folder",
            str(folder),
            "--model",
            "resnet50",
            "--output",
            str(tmp_path / "cnn.csv"),
            "--recursive",
        ],
        "cell-centric": [
            "cell-centric",
            "--folder",
            str(folder),
            "--output",
            str(tmp_path / "cells.csv"),
            "--detector",
            "morphometry",
        ],
        "mil": [
            "mil",
            "--folder",
            str(folder),
            "--output",
            str(tmp_path / "mil.csv"),
        ],
        "cycle-fit": [
            "cycle-fit",
            "--input",
            str(cycle_csv),
            "--output",
            str(tmp_path / "fit.json"),
        ],
        "evaluate": [
            "evaluate",
            "--image-dir",
            str(folder),
            "--output",
            str(tmp_path / "benchmark.json"),
        ],
        "gui": ["gui", "--checkpoint", str(tmp_path / "weights.pt")],
    }

    parsed = {name: parser.parse_args(argv) for name, argv in commands.items()}

    assert set(parsed) == {"classify", "cell-centric", "mil", "cycle-fit", "evaluate", "gui"}
    assert all(namespace.command == name for name, namespace in parsed.items())
    assert parsed["classify"].recursive is True
    assert parsed["cell-centric"].detector == "morphometry"
    assert parsed["evaluate"].models == "cnn,cell-centric,mil"


def test_classify_subcommand_invokes_service_and_exports(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    output = tmp_path / "classified.csv"
    batch = SimpleNamespace(results=[object(), object()])
    service = MagicMock()
    service.classify_folder.return_value = batch
    builder = mocker.patch.object(cli, "_build_cnn_service", return_value=service)

    status = cli.main(
        [
            "classify",
            "--folder",
            str(folder),
            "--model",
            "mobilenet_v2",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--recursive",
        ]
    )

    assert status == 0
    builder.assert_called_once_with("mobilenet_v2", "cpu")
    service.classify_folder.assert_called_once()
    call = service.classify_folder.call_args
    assert call.args == (folder,)
    assert call.kwargs["recursive"] is True
    assert callable(call.kwargs["progress_callback"])
    assert callable(call.kwargs["cancel_flag"])
    service.export_results_csv.assert_called_once_with(batch, output)
    assert "Classified 2 image(s)" in capsys.readouterr().out


def test_cell_centric_subcommand_invokes_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    output = tmp_path / "cells.csv"
    overlays = tmp_path / "overlays"
    pipeline = MagicMock()
    pipeline.process_folder.return_value = [object(), object()]
    pipeline.processing_errors = [(Path("bad.png"), "corrupt")]
    factory = MagicMock(return_value=pipeline)
    fake_module = ModuleType("cycles.stages.cell_centric")
    fake_module.CellCentricPipeline = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cycles.stages.cell_centric", fake_module)

    status = cli.main(
        [
            "cell-centric",
            "--folder",
            str(folder),
            "--output",
            str(output),
            "--detector",
            "yolo",
            "--save-overlays",
            str(overlays),
        ]
    )

    assert status == 0
    factory.assert_called_once_with(detector_mode="yolo")
    pipeline.process_folder.assert_called_once()
    call = pipeline.process_folder.call_args
    assert call.args == (folder,)
    assert call.kwargs["save_overlays_dir"] == overlays
    assert call.kwargs["recursive"] is False
    pipeline.export_results_csv.assert_called_once_with(pipeline.process_folder.return_value, output)
    assert "2 image(s), 1 failed" in capsys.readouterr().out


def test_mil_subcommand_invokes_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    output = tmp_path / "mil.csv"
    heatmaps = tmp_path / "heatmaps"
    batch = SimpleNamespace(results=[object()], failed_images=[])
    pipeline = MagicMock()
    pipeline.process_folder.return_value = batch
    factory = MagicMock(return_value=pipeline)
    fake_module = ModuleType("cycles.stages.mil")
    fake_module.AttentionMILPipeline = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cycles.stages.mil", fake_module)

    status = cli.main(
        [
            "mil",
            "--folder",
            str(folder),
            "--output",
            str(output),
            "--save-heatmaps",
            str(heatmaps),
        ]
    )

    assert status == 0
    factory.assert_called_once_with()
    pipeline.process_folder.assert_called_once()
    call = pipeline.process_folder.call_args
    assert call.args == (folder,)
    assert call.kwargs["save_heatmaps_dir"] == heatmaps
    assert callable(call.kwargs["progress_callback"])
    pipeline.export_results_csv.assert_called_once_with(batch, output)
    assert "Processed 1 image(s), 0 failed" in capsys.readouterr().out


def test_cycle_fit_subcommand_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cycle_csv = tmp_path / "cycle.csv"
    cycle_csv.write_text("day,stage\n1,diestrus\n", encoding="utf-8")
    output = tmp_path / "nested" / "fit.json"
    fit_cyclicity = MagicMock(return_value={"regularity_score": 0.8, "mouse_id": "M7"})
    generate_cycle_plot_data = MagicMock()
    fake_module = ModuleType("cycles.core.cycle")
    fake_module.fit_cyclicity = fit_cyclicity  # type: ignore[attr-defined]
    fake_module.generate_cycle_plot_data = generate_cycle_plot_data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cycles.core.cycle", fake_module)
    monkeypatch.setattr(
        cli,
        "_read_cycle_csv",
        MagicMock(return_value=(["2026-01-01T00:00:00"], ["diestrus"])),
    )

    status = cli.main(
        [
            "cycle-fit",
            "--input",
            str(cycle_csv),
            "--output",
            str(output),
            "--mouse-id",
            "M7",
        ]
    )

    assert status == 0
    fit_cyclicity.assert_called_once_with(
        ["2026-01-01T00:00:00"], ["diestrus"], mouse_id="M7"
    )
    generate_cycle_plot_data.assert_not_called()
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "regularity_score": 0.8,
        "mouse_id": "M7",
    }
    assert "Wrote cycle fit" in capsys.readouterr().out


def test_evaluate_subcommand_invokes_benchmark_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    output = tmp_path / "benchmark.json"
    markdown = tmp_path / "benchmark.md"
    plots = tmp_path / "plots"
    harness = MagicMock()
    factory = MagicMock(return_value=harness)
    fake_module = ModuleType("cycles.eval.benchmark")
    fake_module.BenchmarkHarness = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cycles.eval.benchmark", fake_module)

    status = cli.main(
        [
            "evaluate",
            "--image-dir",
            str(image_dir),
            "--output",
            str(output),
            "--markdown-report",
            str(markdown),
            "--plot-dir",
            str(plots),
            "--models",
            "cnn, cell-centric,mil",
        ]
    )

    assert status == 0
    factory.assert_called_once_with()
    harness.run_benchmark.assert_called_once_with(
        image_dir=image_dir,
        models=["cnn", "cell-centric", "mil"],
        output_json=output,
        markdown_report=markdown,
        plot_dir=plots,
    )
    assert "Wrote benchmark report" in capsys.readouterr().out


def test_gui_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["gui", "--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: cycles gui" in help_text
    assert "--checkpoint" in help_text


def test_main_reports_subcommand_errors(
    tmp_path: Path, mocker: MockerFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    mocker.patch.object(cli, "_build_cnn_service", side_effect=FileNotFoundError("missing weights"))

    status = cli.main(
        [
            "classify",
            "--folder",
            str(folder),
            "--model",
            "missing.pt",
            "--output",
            str(tmp_path / "out.csv"),
        ]
    )

    assert status == 1
    assert "cycles: error: missing weights" in capsys.readouterr().err


def test_cycle_fit_reads_jsonl_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl_path = tmp_path / "predictions.jsonl"
    jsonl_path.write_text(
        json.dumps({
            "sample_id": "d1",
            "day": 1,
            "subject_id": "m1",
            "image_prediction": {"primary_stage": "diestrus"},
            "sequence_prediction": {"final_stage": "diestrus"},
        }) + "\n" +
        json.dumps({
            "sample_id": "d2",
            "day": 2,
            "subject_id": "m1",
            "image_prediction": {"primary_stage": "proestrus"},
            "sequence_prediction": {"final_stage": "proestrus"},
        }) + "\n" +
        json.dumps({
            "sample_id": "d3",
            "day": 3,
            "subject_id": "m1",
            "image_prediction": {"primary_stage": "estrus"},
            "sequence_prediction": {"final_stage": "estrus"},
        }) + "\n" +
        json.dumps({
            "sample_id": "d4",
            "day": 4,
            "subject_id": "m1",
            "image_prediction": {"primary_stage": "metestrus"},
            "sequence_prediction": {"final_stage": "metestrus"},
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "fit.json"

    status = cli.main(["cycle-fit", "--input", str(jsonl_path), "--output", str(output)])

    assert status == 0
    assert output.is_file()
    fit_data = json.loads(output.read_text(encoding="utf-8"))
    assert "regularity_score" in fit_data
    assert "estimated_cycle_length_days" in fit_data


def test_stage_subcommand_invokes_cell_centric_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    output = tmp_path / "stage_out.csv"
    pipeline = MagicMock()
    pipeline.process_folder.return_value = [object()]
    factory = MagicMock(return_value=pipeline)
    fake_module = ModuleType("cycles.stages.cell_centric")
    fake_module.CellCentricPipeline = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cycles.stages.cell_centric", fake_module)

    status = cli.main(
        [
            "stage",
            "--input",
            str(folder),
            "--engine",
            "cell-centric",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    factory.assert_called_once_with(detector_mode="auto")
    pipeline.process_folder.assert_called_once()
    assert "Processed 1 image(s) via Cell-Centric morphometry" in capsys.readouterr().out

