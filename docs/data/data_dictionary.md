# Data Dictionary

This document describes the processed features used in the EMH study.

## Variable Definitions

| Variable | Description | Logic | Dimension |
| :--- | :--- | :--- | :--- |
| **Returns** | Daily log returns of the asset. | $\ln(P_t / P_{t-1})$ | $T \times N$ |
| **Mask** | Binary indicator of valid trading activity. | $(P_t \neq \text{NaN}) \land (\text{Vol}_t > 0)$ | $T \times N$ |
| **Amihud** | Standardized illiquidity measure. | $z\left(\frac{\|R_t\|}{\text{Price}_t \times \text{Vol}_t}\right)$ | $T \times N$ |

## Dataset Metadata

- **Tickers**: `EWZ` (Brazil), `EWW` (Mexico), `ECH` (Chile), `GXG` (Colombia).
- **Frequency**: Daily.
- **Span**: 2015-01-01 to 2024-01-01 (approx. 2260 samples).
- **Missing Value Policy**: Forward-filled prices. Returns zeroed where mask is 0.

## Structural Notes

Data is stored as CSVs in `data/processed/` where each file corresponds to one variable across all tickers.
The index is the date (ISO 8601).
Columns are the ticker symbols.
