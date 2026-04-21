# EMH-LatAm: Efficient Market Hypothesis in Latin America

EMH-LatAm is a reproducible research project for studying the Efficient Market Hypothesis in selected Latin American markets using conditional diffusion models and explainable artificial intelligence.

The repository is designed as an end-to-end scientific workflow that integrates data preparation, time-series-safe evaluation, diffusion-based forecasting, interpretability analysis, and manuscript generation.

## Overview

This project investigates whether selected Latin American equity markets exhibit out-of-sample predictive structure that may be relevant to weak-form market efficiency. The empirical framework focuses on Brazil (EWZ), Mexico (EWW), Chile (ECH), and Colombia (GXG), and evaluates whether diffusion-based models provide robust predictive performance relative to appropriate benchmarks.

Rather than treating predictive performance as direct proof for or against market efficiency, the repository is structured to support careful analysis of statistical significance, robustness across windows and markets, and interpretation of model behavior through explainable AI.

## Research Scope

The project addresses four main objectives:

1. Evaluate whether conditional diffusion models capture predictive structure in Latin American financial time series.
2. Compare diffusion-based forecasts against benchmark statistical and machine learning models under walk-forward validation.
3. Examine whether any observed predictive gains remain stable across countries, periods, and forecast horizons.
4. Use explainable AI methods to identify which temporal patterns or features are most influential in model outputs.

## Methodological Features

The repository includes the following components:

- Conditional diffusion models for financial forecasting
- Walk-forward validation designed for time series
- Causal preprocessing and leakage-aware data handling
- Explainable AI analysis using attribution methods for sequential data
- Statistical testing for model comparison
- Interactive web interface for exploring forecasts and results
- LaTeX manuscript workflow with generated table/figure assets for Overleaf

## Repository Structure

```text
.
|- app/            # Streamlit interface
|- archive/        # Archived legacy notebook artifacts
|- data/           # Raw and processed datasets
|- docs/           # Documentation, theory, and audit notes
|- figures/        # Manuscript-ready figures (generated + static)
|- paper/          # Manuscript and bibliography
|- reports/        # Generated figures, tables, and logs
|- scripts/        # Orchestration + literature tooling
|- tables/         # Manuscript-ready LaTeX tables (generated + static)
|- tex/            # Primary LaTeX manuscript (main.tex + sections)
`- src/            # Core research code (data/model/eval/xai)
````

## Installation

Create the environment and install the required dependencies:

```bash
pip install -e .
```

## Quick Start

### Unified pipeline (recommended)

```bash
python scripts/run_pipeline.py
```

Options:

```bash
python scripts/run_pipeline.py --ingest
python scripts/run_pipeline.py --skip-xai
python scripts/run_pipeline.py --skip-train --skip-experiments
python scripts/run_pipeline.py --skip-manuscript-assets
python scripts/run_pipeline.py --skip-working-paper
```

Build manuscript assets only:

```bash
python scripts/build_manuscript_assets.py
```

Build Overleaf-ready working-paper bundle only:

```bash
python scripts/build_working_paper.py
```

### Step-by-step (manual)

```bash
PYTHONPATH=. python src/data/ingest.py
PYTHONPATH=. python src/data/preprocess.py
PYTHONPATH=. python src/models/trainer.py
PYTHONPATH=. python src/experiments/run_loop.py
PYTHONPATH=. python src/visualization/plot_results.py
PYTHONPATH=. python src/xai/explain.py
```

### Launch the web application

```bash
streamlit run app/main.py
```

## Data and Market Coverage

The current implementation focuses on selected Latin American market proxies:

- Brazil: EWZ
- Mexico: EWW
- Chile: ECH
- Colombia: GXG

All datasets, transformations, and preprocessing decisions should be documented under `docs/data/`.

## Validation and Scientific Standards

This repository is intended for research use and follows a reproducibility-oriented design. Core principles include:

- out-of-sample evaluation only for primary claims,
- walk-forward or expanding-window validation,
- explicit control of randomness and experiment settings,
- separation of exploratory analysis from validated results,
- documentation of assumptions, limitations, and methodological risks.

Claims about market efficiency should be interpreted cautiously and only in relation to the exact data, features, forecast horizon, validation scheme, and benchmark set used in each experiment.

## Results

Baseline experiments are stored under `reports/` and should be interpreted as study-specific empirical findings rather than general conclusions about market efficiency.

A representative summary sentence from the robust implementation is:

> Under expanding-window benchmark-first evaluation, random walk remains best by RMSE across tested \((L,H)\) configurations, while diffusion underperforms simple baselines; this is disciplined negative-result evidence for weak-form predictability tests in this setup.

Figures, tables, attribution analyses, and experiment logs are available in the `reports/` directory.

## Documentation

Project documentation is organized under `docs/` and is expected to cover:

- theoretical foundations of the Efficient Market Hypothesis,
- diffusion models for time-series forecasting,
- explainable AI methodology,
- data provenance and preprocessing,
- experimental design and evaluation protocol,
- reproducibility and audit notes,
- limitations and threats to validity.

## Manuscript Workflow

Primary manuscript entrypoint is `tex/main.tex`. Experimental outputs are transformed into stable manuscript assets by `scripts/build_manuscript_assets.py`, which writes:

- `tables/generated/*.tex`
- `figures/generated/*`

See `asset_pipeline.md` for the full Overleaf-oriented workflow.

## Reproducibility

To ensure reproducibility:

- keep configurations version-controlled,
- record random seeds and environment details,
- avoid modifying processed data manually,
- document failed and exploratory experiments,
- verify results through independent reruns before reporting them.

## Paper Download Pipeline (Scopus + Unpaywall)

This repository now includes a quantity-aware and latest-aware paper downloader:

- `scripts/search_scopus.py`: Scopus metadata search (`title`, `doi`, `year`, `source`, `cited_by`)
- `scripts/download_open_access.py`: DOI download via Unpaywall, optional `scihub-cli` fallback
- `scripts/topic_batch_download.py`: end-to-end search + download workflow

Set credentials in your shell profile:

```bash
export ELSEVIER_API_KEY="<your_elsevier_key>"
export UNPAYWALL_EMAIL="<your_unpaywall_email>"
```

Standard usage:

```bash
python scripts/topic_batch_download.py --keywords "pedestrian simulation" --quantity-mode batch --outdir ./downloads
```

Latest papers (default 3-year window):

```bash
python scripts/topic_batch_download.py --keywords "pedestrian simulation" --quantity-mode batch --latest --outdir ./downloads
```

Explicit count + latest:

```bash
python scripts/topic_batch_download.py --keywords "pedestrian simulation" --target 12 --latest --outdir ./downloads
```

Search-only and DOI-only modes:

```bash
python scripts/search_scopus.py --query 'TITLE-ABS-KEY("pedestrian simulation") AND PUBYEAR > 2022' --count 20 --sort=-coverDate
python scripts/download_open_access.py --doi "10.2307/2392994" --outdir ./downloads --scihub-fallback auto
```

## Citation

If you use this repository in academic work, cite it as:

```bibtex
@software{talero_sarmiento_2026_emh_latam,
  author = {Talero-Sarmiento, Leonardo H.},
  title = {EMH-LatAm: Efficient Market Hypothesis in Latin America via Diffusion Models},
  year = {2026},
  url = {https://github.com/eitt/EMH}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
