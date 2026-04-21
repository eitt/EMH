# Reproducibility Guide

Follow these steps to reproduce the research results.

## 1. Environment Setup
```bash
pip install -e .
```
*Note: Python 3.10+ is recommended.*

## 2. One-Command Pipeline (Recommended)
Run the full workflow from processed data to model outputs:

```bash
python scripts/run_pipeline.py
```

Common options:

```bash
python scripts/run_pipeline.py --ingest
python scripts/run_pipeline.py --skip-xai
python scripts/run_pipeline.py --skip-train --skip-experiments
```

## 3. Manual Step-by-Step Execution
Use this only when debugging a specific stage.

```bash
PYTHONPATH=. python src/data/ingest.py
PYTHONPATH=. python src/data/preprocess.py
PYTHONPATH=. python src/models/trainer.py
PYTHONPATH=. python src/experiments/run_loop.py
PYTHONPATH=. python src/visualization/plot_results.py
PYTHONPATH=. python src/xai/explain.py
```

Best model weights are saved to `reports/logs/best_diffusion_model.pt`.  
Tables and figures are saved under `reports/tables/` and `reports/figures/`.

## 4. Visualization (App)
Launch the interactive playground:

```bash
streamlit run app/main.py
```

## Seed Control
Current scripts do not enforce one global seed entrypoint yet. For strict reproducibility, set seeds in each experiment script and record them in the report metadata.
