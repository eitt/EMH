# EMH Interpretation Note

Date: 2026-04-21

## 1) What this paper tests

This project tests weak-form EMH in a **benchmark-relative forecasting sense**:

- target: out-of-sample cumulative returns over horizon \(H\),
- estimand: forecast-loss difference versus random walk under equal information sets,
- inference: statistical predictability comparisons, not structural market mechanism identification.

## 2) What the empirical design can conclude

The design can conclude whether a candidate model:

1. reduces forecast loss relative to random walk in out-of-sample windows,
2. does so consistently across horizons/lookbacks/windows,
3. passes comparative inference checks (DM tests) under the implemented assumptions.

These are valid statements about **predictive content** in the chosen data/model setup.

## 3) What the empirical design cannot conclude

The design cannot, by itself, conclude:

1. that markets are structurally inefficient in a causal sense,
2. that any model signal is economically exploitable after costs,
3. that predictive gains imply persistent arbitrage opportunities,
4. that XAI feature saliency identifies causal return drivers.

## 4) How forecast superiority should be interpreted

If a model beats random walk robustly, the correct interpretation is:

- evidence against weak-form EMH **conditional on this information set, sample, and validation design**.

If a model fails to beat random walk, the correct interpretation is:

- no robust evidence of incremental predictability beyond simple benchmarks in this setup.

In either case, interpretation must remain conditional and design-specific.

## 5) Why predictive wins are not automatic economic inefficiency

Forecast wins are statistical objects. Economic inefficiency claims require additional layers:

1. implementable trading rule mapping forecasts to positions,
2. realistic turnover and transaction-cost assumptions,
3. risk adjustment and portfolio constraints,
4. out-of-sample economic utility evaluation.

Without these, conclusions should remain at the level of statistical predictability.

## 6) Current project interpretation

Under the revised robust pipeline:

- random walk remains strongest by RMSE across tested configurations,
- diffusion is significantly worse than random walk in DM tests,
- therefore current evidence is consistent with limited extractable short-horizon predictability in this implementation.

This is a methodologically credible negative-result outcome for weak-form EMH testing.
