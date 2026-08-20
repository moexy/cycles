# cycles

`cycles` is a high-performance toolkit for assessing rodent estrous phase from vaginal
cytology images and analyzing longitudinal cycle regularity. It combines deep-learning
predictions with interpretable cellular morphometry, supports both single images and
error-isolated batch processing, and provides command-line and desktop workflows for
training, staging, review, and benchmarking.

The package recognizes the four canonical phases:

- **Diestrus** — leukocyte-predominant cytology.
- **Proestrus** — predominantly nucleated epithelial cells.
- **Estrus** — predominantly cornified squamous cells.
- **Metestrus** — a mixed transitional population, often with cornified cells and
  leukocytes.

Cell-centric analysis also reports **Insufficient Cells** as an explicit quality-control
outcome when fewer than five valid cells are available. This prevents a low-information
slide from being presented as a confident biological stage.

## Staging methods

### CNN image classification

The CNN workflow performs whole-image transfer learning and inference with PyTorch and
Torchvision backbones. Classification results include the predicted stage, calibrated
stage probabilities, confidence and confidence-index values, transition status, and
optional raw logits. Model checkpoints are self-describing: architecture, class order,
input size, training epoch, validation accuracy, creation time, and additional metrics
travel with the learned weights.

Batch inference isolates failures per image, so one unreadable or malformed file does
not discard the remainder of a run.

### Cell-Centric Cytology

The interpretable cytology workflow detects and profiles individual cells before staging
the slide. It distinguishes:

- leukocytes,
- nucleated epithelial cells,
- cornified squamous cells, and
- debris.

Each cell receives morphometric and intensity measurements such as area, perimeter,
circularity, aspect ratio, centroid, bounding box, and intensity statistics. Slide-level
counts and cellular fractions then produce an explainable phase assessment with a
plain-language rationale and transition indication.

Ultralytics YOLO weights can provide learned cell detection when available. If weights
are missing or cannot be loaded, the pipeline remains usable through GPU- or CPU-backed
watershed morphometry built with NumPy, SciPy, and scikit-image.

### Attention-MIL

The Attention Multiple-Instance Learning workflow represents a large slide as a bag of
image patches. A learned attention mechanism weights informative regions and aggregates
them into a slide-level phase prediction. Attention scores can be inspected to understand
which patches contributed most strongly, while patch batching keeps inference practical
for large microscopy images.

## Longitudinal cyclicity tracking

A sequence of dated assessments can be grouped by animal and fitted as an estrous cycle.
The analysis reports:

- an estimated cycle length in days,
- a regularity score,
- deviations from the expected phase progression,
- prolonged consecutive diestrus,
- pseudopregnancy indicators, and
- indexed anomalies with explanatory messages.

The shared `EstrousStage` representation keeps CNN, cell-centric, MIL, GUI, CLI, and cycle
analytics results interoperable.

## Command-line and desktop tools

After installation, two entry points are available:

```text
cycles       Command-line workflows for classification, staging, training, evaluation,
             and longitudinal analysis
cycles-gui   PySide6 desktop application for interactive image and result review
```

The CLI is suitable for reproducible pipelines and batch processing. The GUI supports
interactive selection, visualization, inspection of confidence and cell metrics, and
review of longitudinal results. Matplotlib-based figures can be used in both evaluation
and desktop reporting workflows.

## Benchmarking and evaluation

The evaluation tools compare predicted and reference stages with accuracy, balanced
accuracy, precision, recall, F1 scores, confusion matrices, and per-class summaries.
Benchmark runs measure method-level throughput and latency while preserving failed-image
diagnostics. These reports make it possible to compare CNN, Cell-Centric Cytology, and
Attention-MIL methods on the same dataset and class order.

## Installation

Python 3.11 or newer is required. Install the project and its runtime dependencies with
an environment manager such as `uv`:

```bash
uv sync
```

Install development dependencies when working on the package:

```bash
uv sync --extra dev
```

The main runtime stack includes PyTorch, Torchvision, Pillow, NumPy, SciPy,
scikit-learn, scikit-image, Ultralytics, PySide6, and Matplotlib.

## Python API

Core result types are importable from the top-level package:

```python
from cycles import (
    BatchClassificationResult,
    CellType,
    ClassificationResult,
    CycleFitResult,
    EstrousStage,
    SlideCellMetrics,
    StagingResult,
)

stages = EstrousStage.canonical_stages()
print([stage.display_name for stage in stages])
```

`cycles.core` additionally exposes `CellProfile` and `CheckpointMetadata` for pipelines
that build cellular measurements or write portable checkpoints.

## Intended use

`cycles` is a research tool for rodent vaginal cytology assessment and longitudinal
analysis. Predictions and quality-control flags should be reviewed in the context of the
imaging protocol, staining method, model validation data, and study design; they are not
a substitute for expert review when experimental decisions require manual confirmation.
## Morphology-first local VLM workflow

Apple-Silicon dependencies are optional:

```bash
uv sync --extra mlx
```

Run local inference without sequence context:

```bash
cycles vlm-local --input slides/ --model mlx-community/Qwen3-VL-8B-Instruct-4bit \
  --output runs/vlm/results.jsonl
```

Add a frozen calibrator and the exact `sample_id,image_path,subject_id,day` sequence manifest to
enable guarded adjacent-stage reconciliation. Review the resulting JSONL in the GUI's `VLM Review`
tab; reviews are append-only and teacher export is explicit.

See `docs/superpowers/specs/2026-08-20-estrous-cytology-mlx-v3-design.md` for the blinded annotation,
training, provenance, and acceptance-gate design.
