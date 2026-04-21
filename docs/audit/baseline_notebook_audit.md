# Baseline Notebook Audit: `archive/legacy_notebook/[2026_1]_[Fin]_Difussion_Model_EHM.ipynb`

## Overview
The baseline notebook provides a skeleton for a diffusion-based approach to testing the Efficient Market Hypothesis (EMH) in Latin American markets. However, the current implementation is primarily a **procedural mockup** rather than a functional research pipeline.

## Extracted Logic

### 1. Data Ingestion
- **Source**: Yahoo Finance via `yfinance`.
- **Assets**: `EWZ` (Brazil), `EWW` (Mexico), `ECH` (Chile), `GXG` (Colombia).
- **Timeframe**: 2018-01-01 to 2023-12-31.
- **Frequency**: Daily.

### 2. Feature Engineering
- **Log Returns**: `np.log(pt / pt-1)`.
- **Volume Mask**: Indicators for trading activity.
- **Amihud Illiquidity**: `|Return| / (Volume * Price)`, standardized with a 60-day rolling Z-score.
- **Context Construction**: Concatenates returns, masks, and illiquidity over a lookback window $L$.

### 3. Model Architecture
- **Class**: `DiffusionRegressionNet`.
- **Type**: Multi-Layer Perceptron (MLP) with a time-embedding layer.
- **Input**: Concatenated target dimension + time embedding + flattened context window.
- **Output**: Predicted target returns.

### 4. Interpretation (XAI)
- **Method**: Integrated Gradients (via `Captum`).
- **Goal**: Measure the attribution of historical context features to the model's prediction.
- **EMH Operationalization**: A "Z-score of importance" is calculated. If the maximum Z-score of feature attribution exceeds a threshold (2.5), EMH is "rejected".

## Critical Findings

| Category | Finding | Status |
| :--- | :--- | :--- |
| **Training** | The model is **never trained**. The XAI analysis is performed on randomly initialized weights. | ❌ Critical Failure |
| **Diffusion** | The "diffusion" aspect is purely structural (a time input). No diffusion process (noising/denoising) is implemented. | ⚠️ Placeholder |
| **Validation** | No train/test split or out-of-sample validation. It uses the last 100 samples for attribution check only. | ❌ Methodological Gap |
| **Leakage** | High risk of look-ahead bias if the Amihud standardized window or volume masks are not handled with strict temporal boundaries. | ⚠️ Risk |

## Conclusion
The notebook serves as a **UI/Workflow draft** but lacks any empirical validity. It demonstrates the *intent* to use diffusion and XAI but does not execute the underlying science. The refactored project must implement the actual training loop and a proper diffusion probability model.
