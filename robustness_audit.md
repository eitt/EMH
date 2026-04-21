# Robustness Audit

Date: 2026-04-21

## 1) Major weaknesses found in original pipeline

1. Single holdout split only; no rolling or expanding-window distribution of forecast outcomes.
2. Benchmark set too narrow (random walk, historical mean, ridge, diffusion only).
3. No explicit leakage assertion file; temporal integrity checks were implicit.
4. Target indexing logic was fragile (`range(L, T-H)`), excluding final admissible origin and making horizon handling less transparent.
5. Loss reporting relied mainly on RMSE; weak coverage of MAE/directional behavior.
6. Diffusion stability and ablation evidence were not systematically stored.
7. XAI layer lacked explicit stability diagnostics across seeds/horizons.

## 2) Leakage risks identified

1. Potential overlap risk between terminal train targets and initial validation targets for multi-step horizons.
2. Potential ambiguity around rolling-feature standardization timing.
3. Lack of explicit dataset-level assertions for index ordering, duplicate timestamps, and feature alignment.

## 3) Leakage controls added

1. Strict temporal integrity checks in preprocessing and dataset building:
   - sorted unique time index required,
   - aligned indices/columns across all feature frames.
2. Illiquidity z-score now uses strictly lagged rolling moments (`shift(1)`).
3. Horizon overlap assertions implemented in evaluation loop and exported to:
   - `reports/tables/leakage_checks.csv`
4. Expanding-window generation enforces horizon gap `H-1` between train and validation origins.

## 4) Benchmark weaknesses and fixes

### Before
- Random Walk
- Historical Mean
- Ridge
- Diffusion

### After
- Random Walk
- Historical Mean
- Linear AR baseline
- Ridge (time-aware alpha tuning)
- Shallow MLP baseline
- Random Forest baseline
- Conditional Diffusion

All models now evaluate the same cumulative-return target under shared rolling window protocol.

## 5) Validation weaknesses and fixes

### Before
- one split, one aggregate loss per config.

### After
- expanding-window evaluation,
- per-window loss storage:
  - `reports/tables/per_window_losses.csv`,
- model-rank stability output:
  - `reports/tables/model_rank_stability.csv`,
- leakage assertion output:
  - `reports/tables/leakage_checks.csv`.

## 6) Robustness checks added

1. Multi-loss reporting: RMSE, MAE, directional accuracy.
2. DM tests for benchmark-relative inference (`reports/tables/dm_tests.csv`).
3. Diffusion seed sensitivity (`reports/tables/diffusion_seed_sensitivity.csv`).
4. Diffusion feature ablations (`reports/tables/diffusion_ablation.csv`).
5. Diffusion training stability (`reports/tables/diffusion_training_stability.csv`).
6. XAI seed sensitivity and stability:
   - `reports/tables/xai_seed_sensitivity.csv`
   - `reports/tables/xai_stability.csv`

## 7) Key empirical outcome after robustness upgrades

1. Random walk remains lowest-RMSE model in all tested \((L,H)\) settings.
2. Diffusion underperforms all simple benchmarks materially in this implementation.
3. DM tests show diffusion significantly worse than random walk in all configurations.
4. Some simple baselines are statistically close to random walk in several \(H=5\) settings.

This is a valid negative-result outcome under benchmark-first EMH testing logic.

## 8) Unresolved limitations

1. No ARIMA/ARIMAX, GARCH, or VAR/VECM benchmarks yet.
2. No nested time-series hyperparameter search across all model classes.
3. No transaction-cost-adjusted strategy layer; no economic-significance claims.
4. XAI stability evidence still limited in sample and seed range.
5. Model Confidence Set not yet integrated.

## 9) Recommended next steps

1. Add ARIMA/GARCH benchmark modules with same rolling windows.
2. Add nested tuning for diffusion/MLP across time folds.
3. Add calibration and distributional scoring if probabilistic claims are made.
4. Add turnover/cost-aware strategy evaluation for economic interpretation.
5. Integrate MCS procedure and window-by-window inference robustness.
