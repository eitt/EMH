from __future__ import annotations

import numpy as np


def _loss(y_true: np.ndarray, y_pred: np.ndarray, loss: str = "mse") -> np.ndarray:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of true values and predictions must match.")
    if loss == "mse":
        return (y_true - y_pred) ** 2
    if loss == "mae":
        return np.abs(y_true - y_pred)
    raise ValueError(f"Unsupported loss type: {loss}")


def _pairwise_diff(loss_matrix: np.ndarray) -> np.ndarray:
    # loss_matrix: [n_obs, n_models]
    n_models = loss_matrix.shape[1]
    diffs = np.zeros((n_models, n_models, loss_matrix.shape[0]), dtype=float)
    for i in range(n_models):
        for j in range(n_models):
            diffs[i, j] = loss_matrix[:, i] - loss_matrix[:, j]
    return diffs


def _t_stat(diffs: np.ndarray) -> np.ndarray:
    # diffs: [n_models, n_models, n_obs]
    n_obs = diffs.shape[2]
    mean_diff = np.mean(diffs, axis=2)
    var_diff = np.var(diffs, axis=2, ddof=1)
    se = np.sqrt(var_diff / n_obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.divide(mean_diff, se)
    t[np.isnan(t)] = 0.0
    return t


def _bootstrap_max_t(diffs: np.ndarray, n_bootstrap: int = 500, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_models, _, n_obs = diffs.shape
    bootstrap_max = np.zeros((n_bootstrap,), dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_obs, size=n_obs)
        sampled = diffs[:, :, idx]
        t_star = _t_stat(sampled)
        # Max statistic across all model comparisons.
        bootstrap_max[b] = np.max(np.abs(t_star))
    return bootstrap_max


def compute_model_confidence_set(
    loss_matrix: np.ndarray,
    model_names: list[str],
    alpha: float = 0.05,
    n_bootstrap: int = 500,
    seed: int | None = None,
) -> dict[str, object]:
    """Compute a simplified Model Confidence Set (MCS) for competing models.

    Args:
        loss_matrix: Array of shape [n_obs, n_models] with per-observation loss values.
        model_names: Names of the competing models.
        alpha: Confidence level for exclusion.
        n_bootstrap: Number of bootstrap replications.
        seed: Random seed for bootstrap.

    Returns:
        Dictionary with kept_models, excluded_models, p_values, and mcs_level.
    """
    if loss_matrix.ndim != 2:
        raise ValueError("loss_matrix must be 2D")
    if loss_matrix.shape[1] != len(model_names):
        raise ValueError("Number of model names must match loss matrix columns.")

    remaining = list(model_names)
    remaining_index = list(range(len(model_names)))
    current_loss = loss_matrix.copy()
    p_values = {name: 1.0 for name in model_names}

    while len(remaining) > 1:
        diffs = _pairwise_diff(current_loss)
        t_stats = _t_stat(diffs)

        # Only compare remaining models
        t_stats = t_stats[np.ix_(remaining_index, remaining_index)]
        max_t = np.max(np.abs(t_stats))
        bootstrap_max = _bootstrap_max_t(diffs[np.ix_(remaining_index, remaining_index, np.arange(current_loss.shape[0]))], n_bootstrap=n_bootstrap, seed=seed)
        p_value = float(np.mean(bootstrap_max >= max_t))

        worst_idx = int(np.argmax(np.max(t_stats, axis=1)))
        candidate = remaining[worst_idx]
        p_values[candidate] = p_value

        if p_value < alpha:
            break

        # Exclude the worst model and continue.
        excluded = remaining.pop(worst_idx)
        remaining_index.pop(worst_idx)
        p_values[excluded] = p_value

    kept_models = remaining
    excluded_models = [name for name in model_names if name not in kept_models]

    return {
        "kept_models": kept_models,
        "excluded_models": excluded_models,
        "p_values": p_values,
        "mcs_level": 1.0 - alpha,
    }
