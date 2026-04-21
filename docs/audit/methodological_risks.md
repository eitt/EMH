# Methodological Risks

This document outlines the risks that could invalidate the scientific claims of the study.

## 1. Lack of Training (Immediate Risk)
- **Status**: Critical.
- **Impact**: Currently, the "inefficiency" results are noise-driven.
- **Mitigation**: Implement a complete Diffusion training loop (denoising score matching) with proper loss convergence monitoring.

## 2. Leakage and Look-ahead Bias
- **Risk**: Standardizing data (Amihud) using future information or using data that includes future-dated adjustments.
- **Impact**: Artificial predictive performance.
- **Mitigation**: Use strictly causal transformations. Standardize features using a rolling window that ends *before* the forecast target start time.

## 3. Data Leakage (Stationarity)
- **Risk**: Financial markets are non-stationary. Training on 2018-2021 and testing on 2022 might fail due to regime shifts (e.g., inflation spikes in LatAm).
- **Impact**: Poor generalization or "over-claiming" based on a specific regime.
- **Mitigation**: Use rolling-window backtesting (Walk-forward validation) to evaluate performance across different regimes.

## 4. Conflating Prediction with Inefficiency
- **Risk**: Significant predictability might reflect risk premiums rather than market inefficiency.
- **Impact**: Incorrect theoretical conclusion regarding EMH.
- **Mitigation**: Include a dedicated "EMH Interpretation Layer" that considers transaction costs and risk-adjusted returns before claiming "market inefficiency".

## 5. XAI Instability
- **Risk**: Integrated Gradients can be sensitive to the choice of "baseline" (noise vs. zero).
- **Impact**: Misleading feature importance.
- **Mitigation**: Test attribution stability across multiple baselines and noise realizations.
