"""
Economic significance evaluation for diffusion forecasts.
Computes PnL under transaction costs and Sharpe ratios.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_portfolio_pnl(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    transaction_cost: float = 0.001,  # 10 bps
    initial_capital: float = 1.0,
) -> dict[str, float]:
    """
    Compute portfolio PnL from predictions.
    Assumes equal-weight portfolio based on prediction signs.
    """
    n_assets = predictions.shape[1]
    weights = np.sign(predictions) / n_assets  # Equal weight per asset
    # Transaction costs: cost when changing positions
    # For simplicity, assume we rebalance fully each period
    # Cost = transaction_cost * |weight_change|
    # But since we start from zero, first period cost = transaction_cost * |weight|
    pnl = initial_capital
    prev_weights = np.zeros_like(weights[0])
    for t in range(len(predictions)):
        current_weights = weights[t]
        cost = transaction_cost * np.sum(np.abs(current_weights - prev_weights))
        pnl *= 1 + np.dot(current_weights, actual_returns[t]) - cost
        prev_weights = current_weights
    sharpe = (pnl - initial_capital) / (np.std(np.diff(np.log(np.maximum(pnl, 1e-8)))) + 1e-8)  # Approximate Sharpe
    return {
        "final_pnl": float(pnl),
        "total_return": float(pnl - initial_capital),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(compute_max_drawdown(np.array([pnl]))),  # Placeholder
    }


def compute_max_drawdown(portfolio_values: np.ndarray) -> float:
    """Compute maximum drawdown."""
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    return np.min(drawdown)


def evaluate_economic_significance(
    predictions_df: pd.DataFrame,
    actual_returns_df: pd.DataFrame,
    transaction_costs: list[float] = [0.0001, 0.001, 0.01],  # 1bps, 10bps, 100bps
) -> pd.DataFrame:
    """
    Evaluate economic significance across different transaction cost levels.
    """
    results = []
    for tc in transaction_costs:
        # Assume predictions_df has columns for each model/asset
        # For simplicity, assume one model, predictions are [T, N]
        # But since it's df, perhaps group by model
        # Placeholder: assume predictions_df has 'model', 'asset', 'prediction', 'actual'
        # But to match, perhaps compute per model
        for model in predictions_df['model'].unique():
            model_preds = predictions_df[predictions_df['model'] == model]
            # Pivot to [T, N]
            pred_matrix = model_preds.pivot(index='date', columns='asset', values='prediction').values
            actual_matrix = actual_returns_df.values  # Assume aligned
            pnl_stats = compute_portfolio_pnl(pred_matrix, actual_matrix, transaction_cost=tc)
            results.append({
                "model": model,
                "transaction_cost": tc,
                **pnl_stats,
            })
    return pd.DataFrame(results)