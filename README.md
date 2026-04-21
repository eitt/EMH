# EMH-LatAm: Efficient Market Hypothesis in Latin America

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Framework](https://img.shields.io/badge/framework-PyTorch-orange)](https://pytorch.org/)

A professional research pipeline for studying **Efficient Market Hypothesis (EMH)** in emerging Latin American markets using **Conditional Diffusion Models** and **Explainable AI**.

## 🔬 Project Overview
This repository implements an end-to-end scientific workflow to investigate market efficiency in **Brazil (EWZ)**, **Mexico (EWW)**, **Chile (ECH)**, and **Colombia (GXG)**. By reformulating return prediction as a conditional denoising task, we quantify the historical structure present in financial signals.

### Key Features
- **Diffusion-Based Forecasting**: Uses a Conditional Gaussian Diffusion process to model future return distributions.
- **Explainable AI (XAI)**: Integrated Gradients attribution to identify predictive "footprints" in historical data.
- **Scientific Rigor**: Causal preprocessing, walk-forward validation, and Z-score based EMH hypothesis testing.
- **Interactive Playground**: Streamlit-based interface for real-time visualization and experiment browsing.
- **Manuscript Workflow**: Automated Quarto scaffold for research paper generation.

## 🚀 Quickstart

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Ingest & Preprocess
```powershell
$env:PYTHONPATH="."
python src/data/ingest.py
python src/data/preprocess.py
```

### 3. Model Training & XAI
```powershell
python src/models/trainer.py
python src/xai/explain.py
```

### 4. Launch Playground
```powershell
streamlit run app/main.py
```

## 📊 Result Highlights
Our baseline experiments (2015-2024) yielded a **Max Z-Score of 3.18**, indicating a rejection of weak-form efficiency within the specified parameters. Detailed heatmaps are available in `reports/figures/xai/`.

## 📂 Repository Structure
- `src/`: Modular research code (data, models, xai, stats).
- `docs/`: Theoretical foundations and audit reports.
- `app/`: Streamlit web application.
- `paper/`: Manuscript scaffold and citation metadata.
- `reports/`: Aggregated results, logs, and figures.

## 📜 Citation
If you use this work in your research, please cite:
```bibtex
@software{emh_latam_2026,
  author = {Antigravity},
  title = {EMH-LatAm: Efficient Market Hypothesis in Latin America via Diffusion Models},
  year = {2026},
  url = {https://github.com/eitt/EMH}
}
```
