# Reproducibility Gaps

The following gaps must be addressed to ensure the research is reproducible and scientifically valid:

## 1. Environment and Dependencies
- **Problem**: No `requirements.txt` or `environment.yml` existed. Library versions (especially `torch`, `captum`, and `yfinance`) are unknown.
- **Fix**: Create `pyproject.toml` and `requirements.txt` with pinned versions.

## 2. Stochastic Control
- **Problem**: Random seeds are not set for `numpy`, `torch`, or the diffusion noise generation. Consecutive runs will yield different "Z-scores" even if the model were trained.
- **Fix**: Implement a global `seed_everything` utility and log seeds in the experiment registry.

## 3. Data Provenance
- **Problem**: `yfinance` data is dynamic. Historical adjustments (dividends, splits) can change data points retroactively.
- **Fix**: Implement a local caching mechanism for raw data with versioning (`data/raw/v1_...csv`).

## 4. Execution State
- **Problem**: The notebook relies on "hidden state" (e.g., variables like `ret`, `msk` defined in previous runs).
- **Fix**: Refactor into functional pipelines where each step is idempotent and clearly takes inputs/outputs.

## 5. Model Weights
- **Problem**: Predictions are made on un-saved, un-trained weights.
- **Fix**: Implement a model checkpointing system in `reports/logs/`.
