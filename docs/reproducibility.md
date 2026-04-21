# Reproducibility Guide

Follow these steps to reproduce the research results.

## 1. Environment Setup
```bash
pip install -r requirements.txt
```
*Note: Python 3.10+ is recommended.*

## 2. Data Ingestion
Download the latest market data:
```bash
$env:PYTHONPATH="."
python src/data/ingest.py
```
This saves raw data to `data/raw/` and a symlink to `latest_raw.csv`.

## 3. Preprocessing
Clean and transform the data:
```bash
python src/data/preprocess.py
```

## 4. Modeling (Training)
Train the Conditional Diffusion Model:
```bash
python src/models/trainer.py
```
Best weights are saved to `reports/logs/best_diffusion_model.pt`.

## 5. Explainability (XAI)
Generate feature importance and EMH Z-scores:
```bash
python src/xai/explain.py
```
Figures are saved to `reports/figures/xai/`.

## 6. Visualization (App)
Launch the interactive playground:
```bash
streamlit run app/main.py
```

## Seed Control
All stochastic processes (diffusion noise, model initialization) are seeded in the `src/utils/` modules (to be implemented/finalized) to ensure consistency across runs.
