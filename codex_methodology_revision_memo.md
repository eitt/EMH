# Codex Methodology Revision Memo

Date: 2026-04-21

## 1. Major Methodological Weaknesses in the Original Draft
1. The manuscript was a placeholder skeleton with unsupported claims, missing identification logic, and no defensible mapping from model output to weak-form EMH inference.
2. References were not managed through a real `.bib` file; citation handling was broken.
3. The empirical section overclaimed diffusion/XAI evidence without benchmark-disciplined out-of-sample comparison.
4. Data construction, target definition, and split logic were underspecified.
5. The train/validation boundary for multi-day horizons allowed target-window overlap, contaminating holdout evaluation for $H>1$.
6. Descriptive diagnostics (stationarity, dependence, heavy tails) were absent from the paper narrative.

## 2. Corrections Implemented
1. Rewrote `paper/manuscript.qmd` into full scientific structure:
   - Introduction
   - Literature positioning and gap
   - Data and variable construction
   - Econometric strategy and identification
   - Preliminary empirical evidence
   - Robustness agenda
   - Conclusion
2. Added a proper bibliography file at `paper/references.bib` and aligned all in-text citations with used references only.
3. Formalized notation and equations for returns, Amihud construction, horizon target, diffusion objective, RMSE, and Diebold-Mariano inference.
4. Enforced horizon-safe holdout splitting in code (`src/models/evaluation/data_loader.py`) by introducing a validation gap of `H-1` observations.
5. Re-ran experiment loop and regenerated result artifacts:
   - `reports/tables/experiment_results.csv`
   - `reports/tables/rmse_pivot.csv`
   - `reports/tables/dm_tests.csv`
   - `reports/figures/rmse_comparison.png`
   - `reports/figures/pvalue_heatmap.png`
6. Added reproducible diagnostic tables:
   - `reports/tables/summary_statistics.csv`
   - `reports/tables/diagnostic_tests.csv`
   - `reports/tables/return_correlations.csv`
   - `reports/tables/xai_seed_sensitivity.csv`

## 3. Literature Gap Identified (Scopus-Oriented Logic)
The revised gap is precise and defensible:

- Existing Latin American efficiency evidence is largely test-battery based (variance-ratio / nonlinear dependence / adaptive-market framing), with limited benchmark-first multivariate forecast design.
- Core forecasting literature emphasizes out-of-sample benchmark discipline and instability risk in return predictability.
- Diffusion-based financial forecasting is emerging, but high-quality peer-reviewed evidence directly tied to weak-form efficiency testing in Latin American ETF panels is still limited.

Operational gap statement used in manuscript:

There is limited evidence on whether a conditional diffusion forecaster can deliver incremental out-of-sample predictive accuracy over random-walk benchmarks in Latin American ETF returns under explicit, horizon-consistent forecast evaluation.

## 4. Unresolved Limitations
1. Single holdout split remains; no rolling-origin or expanding-window evaluation yet.
2. Diffusion hyperparameters are not systematically tuned via nested validation.
3. Economic significance (trading rule, turnover, transaction-cost net performance) is not yet evaluated.
4. Benchmark set excludes canonical volatility/time-series models (ARIMA/GARCH/VAR).
5. DM inference uses asymptotic normal approximation only; no MCS or alternative loss-function robustness yet.

## 5. Recommended Next Empirical Steps
1. Implement rolling-origin evaluation across multiple market regimes and report dispersion of RMSE/DM outcomes.
2. Add ARIMA/GARCH and simple VAR/VECM baselines where admissible.
3. Add loss-function robustness (`MAE`, `QLIKE`) and model confidence set inference.
4. Add economic significance layer: directional accuracy, turnover-aware strategy simulation, and transaction-cost stress tests.
5. Add forecast calibration diagnostics (distributional calibration and tail behavior) for diffusion outputs.
6. Reassess XAI only after predictive model quality improves; otherwise keep XAI strictly exploratory.
