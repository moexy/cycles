# cyclonaut

Automated rodent estrous phase staging and longitudinal cyclicity analysis from vaginal cytology images.

## Installation

```bash
uv sync --extra mlx
```

## Quickstart

```bash
# Stage a slide or longitudinal subject folder and plot the cycle timeline:
uv run cyclonaut stage \
  --input path/to/images/ \
  --output runs/results.jsonl \
  --csv runs/results.csv \
  --plot runs/timeline.png

# Launch the desktop review GUI:
uv run cyclonaut-gui
```

