# `cycles` — Rodent Estrous Phase Assessment & Tracking Platform

**Repository:** `https://github.com/moexy/cycles.git`  
**Date:** 2026-08-20  
**Status:** All 105 unit/integration tests passing (`100% green`), `ruff check .` clean (0 errors), validated end-to-end on high-resolution OME-TIFF slides and the multi-lab 13,625-image EstrousBank benchmark corpus.

---

## 1. Project Overview & Architecture

`cycles` is a modern, modular Python 3.12+ platform for automated rodent vaginal cytology analysis, multi-modal estrous cycle phase assessment (`diestrus`, `proestrus`, `estrus`, `metestrus`, and `insufficient_cells` QC), longitudinal cyclicity fitting, and visual instruction tuning dataset preparation.

```
cycles/
├── pyproject.toml                     # uv / hatchling build configuration
├── README.md                          # Repository overview & quickstart
├── HANDOFF.md                         # Authoritative engineering handoff & benchmark documentation
├── cycles/
│   ├── __init__.py                    # Package exports & version (0.1.0)
│   ├── core/                          # Shared domain primitives & infrastructure
│   │   ├── types.py                   # Enums (EstrousStage, CellType), dataclasses with slots
│   │   ├── models.py                  # Backbone registry (ResNet-50, InceptionV3, VGG19, MobileNetV2, ConvNeXt),
│   │   │                              # layer freezing (freeze_layers), device selection (MPS, CUDA, CPU),
│   │   │                              # self-describing checkpoint serialization & metadata
│   │   ├── preprocessing.py           # Image discovery, RGB loading, CLAHE/percentile luminance normalization,
│   │   │                              # aggressive cytology stain transforms (ColorJitter, Rotation90, Crop)
│   │   └── cycle.py                   # Markov transition matrix, confidence index (P(1) - P(2)),
│   │                                  # transition anomaly detection, pseudopregnancy detection (>=10d diestrus),
│   │                                  # circular periodogram cyclicity fitting & plot data generator
│   ├── stages/                        # Staging engines & inference pipelines
│   │   ├── cnn.py                     # Classical CNN service (error-isolated batch inference, progress callbacks,
│   │   │                              # cooperative cancellation, CSV/JSON export, CNNTrainerService)
│   │   ├── cell_centric/              # Explainable single-cell cytology staging framework
│   │   │   ├── detector.py            # Scale-aware tiled YOLOv8 + OpenCV multi-scale cytomorphology (sub-100ms)
│   │   │   ├── classifier.py          # Cytomorphological typing (Leukocytes, Nucleated, Cornified, Debris)
│   │   │   ├── staging.py             # Calibrated proportion-space centroids, rules, <5 cell QC guardrail
│   │   │   └── pipeline.py            # Slide-level cell-centric inference & color-coded overlay rendering
│   │   ├── mil/                       # Foundation Model + Multiple Instance Learning for WSI
│   │   │   ├── encoder.py             # ConvNeXt-Tiny 512-dim patch embedding extractor
│   │   │   ├── model.py               # Gated Attention-MIL slide-level classifier with 1D normalized attention
│   │   │   ├── patching.py            # High-resolution slide grid tiling & tissue background filtering
│   │   │   ├── trainer.py             # MIL training loop with early stopping & cosine scheduling
│   │   │   └── pipeline.py            # End-to-end slide inference & diagnostic attention heatmap export
│   │   └── vlm.py                     # Multimodal Vision-Language Model (VLM) interpretation service
│   │                                  # (LLaVA/OpenAI/Gemini/Ollama prompt formatting, structured JSON output)
│   ├── eval/                          # Benchmarking, evaluation, and reporting
│   │   ├── metrics.py                 # Accuracy, Balanced Accuracy, Cohen's Kappa, Macro/Weighted F1,
│   │   │                              # Latency, Throughput, Confusion Matrix & Model Comparison bar charts
│   │   └── benchmark.py               # Multi-model comparative benchmark harness emitting CSV, JSON, Markdown, PNGs
│   ├── cli/                           # Unified command-line interface
│   │   └── main.py                    # Subcommands: classify, cell-centric, mil, cycle-fit, evaluate, train, gui
│   └── gui/                           # Interactive desktop application (PySide6)
│       ├── app.py                     # Safe GUI launcher with headless fallback
│       ├── main_window.py             # 3-pane workbench (toolbar, file list, image canvas, detail bars, timeline)
│       │                              # with 'Show/Hide Labels' toggle (L hotkey) & cooperative cancellation
│       ├── workers.py                 # QThread background worker with progress & cancel support
│       └── canvas.py                  # Matplotlib/Qt interactive canvas for overlays, heatmaps, and cycle timeline
├── scripts/
│   ├── train_resnet50.py              # Standalone multi-worker ResNet-50 fine-tuning script
│   └── create_webdataset_shards.py    # Parallel WebDataset (.tar) sharder with WebP 80 compression & LLaVA JSON
└── tests/                             # Comprehensive pytest test suite (105 tests, 100% green)
    ├── test_types.py                  # Domain enums, properties, slots dataclasses
    ├── test_preprocessing.py          # Discovery, RGB loading, luminance normalization, transforms
    ├── test_models.py                 # Backbone registry, device detection, layer freezing, checkpoints
    ├── test_cycle.py                  # Transition matrix, confidence index, anomalies, pseudopregnancy, fitting
    ├── test_cnn.py                    # CNN inference, cancellation, batch exports, trainer step
    ├── test_cell_centric.py           # Morphology, YOLO verification, staging rules, <5 cell QC
    ├── test_mil.py                    # Patching, ConvNeXt encoder, Gated Attention-MIL, heatmaps
    ├── test_vlm.py                    # VLM prompt formatting, base64 encoding, credential handling, API parsing
    ├── test_eval.py                   # Metrics calculation, confusion matrix plots, benchmark harness
    ├── test_cli.py                    # CLI parser and all subcommands
    └── test_gui_smoke.py              # Headless PySide6 GUI initialization, canvas, worker thread cancellation
```

---

## 2. Key Checkpoints & Available Weights

| Checkpoint | Path | Size | Architecture | Performance & Use Case |
| :--- | :--- | :---: | :--- | :--- |
| **EstrousBank Fine-Tuned ResNet-50** | `runs/resnet50_estrousbank_finetuned.pt` | 205 MB | ResNet-50 (`IMAGENET1K_V2` base, `layer4` + `fc` tuned) | **72.2% Val Acc / 71.3% Test Acc** across 13,625 multi-lab images. Stain-robust (30 ms/slide). |
| **EstrousNet Pretrained ResNet-50** | `/Volumes/SSD/code/EstrousNet/estrousnet_resnet50.pt` | 90 MB | ResNet-50 | Pretrained on legacy Crystal Violet / Methylene Blue dataset (30 ms/slide). |
| **ODES YOLOv8 Cytology Weights** | `/Volumes/SSD/code/EstrousNet/ODES_Object_Detection_For_Estrous_Staging/ODES/finalweight.pt` | 50 MB | Ultralytics YOLOv8 DetectionModel | Trained on rodent vaginal cytology cell crops (Leukocyte, Cornified, Nucleated). |
| **Gated Attention-MIL Head** | `/Volumes/SSD/code/EstrousNet/runs/estrousbank_mil_sota/attention_mil_best.pt` | 0.88 MB | Gated Attention-MIL (512-dim ConvNeXt embeddings) | WSI slide-level classification with diagnostic attention heatmaps. |

---

## 3. WebDataset Shards (`/Volumes/SSD/Bioinformatics/shards/`)

All **13,624 images** from `/Volumes/SSD/Bioinformatics/EstrousBank_Work` have been converted to **WebP 80** and packaged into standard **WebDataset (`.tar`) shards** for multimodal VLM / LLM training and high-throughput PyTorch streaming:

* **Location:** `/Volumes/SSD/Bioinformatics/shards/`
  * `train/`: 11 shards (`train-000000.tar` to `train-000010.tar`, 10,900 samples, ~3.49 GB)
  * `val/`: 2 shards (`val-000000.tar` to `val-000001.tar`, 1,362 samples, ~445 MB)
  * `test/`: 2 shards (`test-000000.tar` to `test-000001.tar`, 1,362 samples, ~433 MB)
* **Sample Structure:**
  * `{key}.webp`: WebP-encoded cytology image (quality 80)
  * `{key}.json`: Metadata + LLaVA / ShareGPT conversational format for visual instruction tuning
  * `{key}.cls`: Integer class index (`0`: diestrus, `1`: proestrus, `2`: estrus, `3`: metestrus)
  * `{key}.txt`: Natural language caption describing stage, stain, magnification, species, and strain

##### Streaming with PyTorch WebDataset:
```python
import webdataset as wds

dataset = (
    wds.WebDataset("/Volumes/SSD/Bioinformatics/shards/train/train-{000000..000010}.tar")
    .shuffle(1000)
    .decode("pil")
    .to_tuple("webp", "json", "cls")
)
for image, metadata, label in dataset:
    print(image.size, metadata["stage"], label)
```

---

## 4. Key Workflows & CLI Commands

### A. Fine-Tuning ResNet-50 (`cycles train`)
```bash
uv run python -m cycles.cli.main train \
  --train-dir "/Volumes/SSD/Bioinformatics/EstrousBank_Work/splits/train" \
  --val-dir "/Volumes/SSD/Bioinformatics/EstrousBank_Work/splits/val" \
  --epochs 15 \
  --batch-size 64 \
  --lr 1e-4 \
  --output "runs/resnet50_estrousbank_finetuned.pt" \
  --device auto
```

### B. High-Speed Cell-Centric Staging (`cycles cell-centric`)
```bash
uv run python -m cycles.cli.main cell-centric \
  --folder "/Volumes/SSD/Imaging/Cycles/samples/batch_5/mouse5" \
  --detector morphometry \
  --save-overlays runs/overlays \
  --output runs/cellcentric_results.csv
```
* **Performance:** Sub-100ms per $2880 \times 2048$ slide on Apple Silicon.
* **Physical Cell Size Invariants:**
  * **Debris / Dust:** $< 35\text{ px}^2$ (filtered out).
  * **Leukocytes:** $35–300\text{ px}^2$ (compact solid dark disc, $N:C > 0.70$).
  * **Nucleated Epithelial:** $350–2000\text{ px}^2$ (round/oval with central nucleus and cytoplasm halo).
  * **Cornified Squamous:** $\ge 2000\text{ px}^2$ (flat polygonal anucleate cell bodies/sheets).

### C. Longitudinal Cycle Tracking (`cycles cycle-fit`)
```bash
# Export fit parameters, regularity score, and pseudopregnancy detection to JSON
uv run python -m cycles.cli.main cycle-fit \
  --input runs/cellcentric_results.csv \
  --output runs/cycle_fit.json

# Generate timeline progression PNG (Diestrus=Blue, Proestrus=Green, Estrus=Red, Metestrus=Orange)
uv run python -m cycles.cli.main cycle-fit \
  --input runs/cellcentric_results.csv \
  --output runs/cycle_timeline.png
```

### D. Attention-MIL WSI Staging (`cycles mil`)
```bash
uv run python -m cycles.cli.main mil \
  --folder "/Volumes/SSD/Imaging/Cycles/samples/batch_5/mouse5" \
  --save-heatmaps runs/heatmaps \
  --output runs/mil_results.csv
```

### E. Unified Multi-Model Benchmarking (`cycles evaluate`)
```bash
uv run python -m cycles.cli.main evaluate \
  --image-dir "/Volumes/SSD/Bioinformatics/EstrousBank_Work/splits/test" \
  --models "cnn,cell-centric,mil" \
  --output runs/benchmark_report.json \
  --markdown-report runs/benchmark_report.md \
  --plot-dir runs/benchmark_plots
```

### F. Interactive PySide6 Desktop GUI (`cycles gui` or `cycles-gui`)
```bash
uv run cycles-gui
```
* Interactive 3-pane workbench with **"Show / Hide Labels"** toggle (hotkey **`L`**), zoomable canvas, live stage probability bars, transition warnings, cell counts, and timeline plot.

---

## 5. Verification & Test Suite

* **Linter:** `uv run ruff check .` passed with **0 errors and 0 warnings**.
* **Test Suite:** `uv run pytest -v` passed all **105 unit and integration tests** in `3.76s`.
* **Git Status:** Fully committed and synchronized with remote repository at `https://github.com/moexy/cycles.git` (`origin/main`).
