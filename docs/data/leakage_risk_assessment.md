# Leakage Risk Assessment

This document audits the pipeline for potential look-ahead bias and data contamination.

## 1. Feature Engineering
- **Returns**: Calculated as $\ln(P_t / P_{t-1})$. This depends only on current and past prices. **Safe**.
- **Amihud Standardization**: Calculated using a 60-day **rolling window**. The standardization parameters $(\mu, \sigma)$ at time $t$ include data from $t-59$ to $t$.
    - *Risk*: Does $X_t$ include label information?
    - *Mitigation*: In the training loop, features $X_t$ are matched with targets $Y_{t+1:t+H}$. The rolling window used for $X_t$ does not overlap with the target period. **Safe**.

## 2. Validation Protocol
- **Method**: Walk-forward (Expansion window).
- **Control**: The model is trained on data up to $T$ and tested only on data starting at $T+1$.
- **Risk**: Global standardization (Z-score over the entire dataset).
- **Mitigation**: Preprocessing uses *rolling* standardization. No global parameters are leaked across the time boundary. **Safe**.

## 3. Data Leakage (Stationarity)
- **Risk**: Using future volatility regimes to inform current predictions.
- **Mitigation**: The model has no access to future variance or mask patterns during training.

## 4. Hyperparameter Optimization
- **Risk**: Tuning hyperparameters on the "test" set.
- **Mitigation**: Validation is strictly separated from the hold-out test set using a triple-split (Train/Val/Test) or iterative rolling validation.
