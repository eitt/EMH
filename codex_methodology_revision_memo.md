# Codex Methodology Revision Memo

Date: 2026-04-21

## 1) Major methodological weaknesses in the prior version

1. Single-split evaluation with limited robustness against time-regime variation.
2. Narrow benchmark family (insufficiently strong comparators for diffusion).
3. Weakly documented leakage controls and no exported leakage-check artifact.
4. Limited loss-function coverage (RMSE-centric reporting).
5. Diffusion stability and ablation evidence not systematically reported.
6. XAI interpretation lacked explicit stability diagnostics.

## 2) Corrections implemented

1. Replaced single split with expanding-window evaluation and horizon gaps.
2. Expanded benchmark family to:
   - Random Walk
   - Historical Mean
   - Linear AR
   - Ridge
   - MLP
   - Random Forest
   - Diffusion
3. Added explicit temporal-integrity checks in preprocessing and sample construction.
4. Added leakage assertions and exported `reports/tables/leakage_checks.csv`.
5. Added multi-metric evaluation (RMSE, MAE, directional accuracy) and per-window storage.
6. Retained DM inference and expanded summaries to multi-model robust comparisons.
7. Added diffusion robustness outputs:
   - `diffusion_training_stability.csv`
   - `diffusion_seed_sensitivity.csv`
   - `diffusion_ablation.csv`
8. Added XAI robustness outputs:
   - `xai_seed_sensitivity.csv`
   - `xai_stability.csv`
9. Updated manuscript architecture and reporting flow to align with robust pipeline outputs.
10. Added Overleaf bundle builder (`scripts/build_working_paper.py`) with article-style checks.

## 3) Updated literature-gap statement

The operational gap addressed is not lack of EMH testing per se, but lack of a benchmark-disciplined, leakage-audited, expanding-window forecast-comparison framework for LatAm ETF weak-form predictability where diffusion models are evaluated fairly against strong simple and nonlinear baselines.

## 4) Current empirical interpretation after robustness upgrades

1. Random walk remains best by RMSE across all tested \((L,H)\) settings.
2. Diffusion is statistically inferior to random walk in all DM comparisons in this implementation.
3. Some simple alternatives are close to random walk in selected settings but not robustly superior.
4. XAI outputs are unstable enough to be treated as model-reliance diagnostics only.

This is a credible negative-result finding under benchmark-first weak-form EMH testing logic.

## 5) Unresolved limitations

1. ARIMA/GARCH/VAR-class benchmarks are still absent.
2. Nested hyperparameter validation remains limited.
3. No transaction-cost/economic-value layer yet.
4. Model Confidence Set not integrated.
5. XAI stability evidence remains sample-limited.

## 6) Recommended next steps

1. Add ARIMA/GARCH/VAR modules under the same window protocol.
2. Implement nested time-series tuning for diffusion and ML baselines.
3. Add economic-significance tests with turnover and transaction costs.
4. Add MCS and additional loss functions (e.g., QLIKE when relevant).
5. Expand regime-specific reporting and post-2023 validation when data are updated.
