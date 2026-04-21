Here is a more professional version of the README, with no icons, no badges, and a more academic tone. I also softened the empirical claim so it reads like a research repository rather than a promotional page.

````md
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
- Quarto-based manuscript workflow for paper preparation

## Repository Structure

```text
.
├── app/            # Web application for interactive exploration
├── configs/        # Configuration files for data, models, and experiments
├── data/           # Raw, interim, and processed datasets
├── docs/           # Documentation, theory, methodology, and audit notes
├── notebooks/      # Research notebooks and archived baseline work
├── paper/          # Manuscript source, references, and appendices
├── reports/        # Generated figures, tables, logs, and model summaries
├── src/            # Modular source code for data, models, evaluation, and XAI
└── tests/          # Unit and integration tests
````

## Installation

Create the environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

If the project is configured through `pyproject.toml`, install it in editable mode:

```bash
pip install -e .
```

## Quick Start

### Data ingestion and preprocessing

```bash
PYTHONPATH=. python src/data/ingest.py
PYTHONPATH=. python src/data/preprocess.py
```

### Model training

```bash
PYTHONPATH=. python src/models/trainer.py
```

### Explainability analysis

```bash
PYTHONPATH=. python src/xai/explain.py
```

### Launch the web application

```bash
streamlit run app/main.py
```

## Data and Market Coverage

The current implementation focuses on selected Latin American market proxies:

* Brazil: EWZ
* Mexico: EWW
* Chile: ECH
* Colombia: GXG

All datasets, transformations, and preprocessing decisions should be documented in the corresponding files under `docs/data/` and `configs/data/`.

## Validation and Scientific Standards

This repository is intended for research use and follows a reproducibility-oriented design. Core principles include:

* out-of-sample evaluation only for primary claims,
* walk-forward or expanding-window validation,
* explicit control of randomness and experiment settings,
* separation of exploratory analysis from validated results,
* documentation of assumptions, limitations, and methodological risks.

Claims about market efficiency should be interpreted cautiously and only in relation to the exact data, features, forecast horizon, validation scheme, and benchmark set used in each experiment.

## Results

Baseline experiments are stored under `reports/` and should be interpreted as study-specific empirical findings rather than general conclusions about market efficiency.

A representative summary sentence may be stated as follows:

> Under the current experimental specification, the baseline results suggest evidence of out-of-sample predictive structure in some market-period settings. These findings require verification through robustness checks, benchmark comparisons, and sensitivity analysis before being interpreted in relation to weak-form efficiency.

Figures, tables, attribution analyses, and experiment logs are available in the `reports/` directory.

## Documentation

Project documentation is organized under `docs/` and is expected to cover:

* theoretical foundations of the Efficient Market Hypothesis,
* diffusion models for time-series forecasting,
* explainable AI methodology,
* data provenance and preprocessing,
* experimental design and evaluation protocol,
* reproducibility and audit notes,
* limitations and threats to validity.

## Manuscript Workflow

The `paper/` directory contains the manuscript scaffold and supporting assets for academic writing. The repository is structured so that experimental outputs can be incorporated into publication-ready tables and figures with minimal manual intervention.

## Reproducibility

To ensure reproducibility:

* keep configurations version-controlled,
* record random seeds and environment details,
* avoid modifying processed data manually,
* document failed and exploratory experiments,
* verify results through independent reruns before reporting them.

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

```

