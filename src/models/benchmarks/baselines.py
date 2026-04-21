from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_context(x: np.ndarray | torch.Tensor) -> np.ndarray:
    x_np = to_numpy(x)
    return x_np.reshape(x_np.shape[0], -1)


class BaselineModel:
    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        return None

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        raise NotImplementedError


class RandomWalkModel(BaselineModel):
    """No-change benchmark for cumulative return targets."""

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        x_np = to_numpy(x)
        n_samples = x_np.shape[0]
        n_assets = x_np.shape[2]
        return np.zeros((n_samples, n_assets), dtype=float)


class HistoricalMeanModel(BaselineModel):
    """Mean lagged return in lookback window for each asset."""

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        x_np = to_numpy(x)
        # Feature 0 is return.
        return x_np[:, :, :, 0].mean(axis=1)


class LinearAutoregressiveModel(BaselineModel):
    """
    Linear AR baseline using only lagged returns per asset.
    Each asset has its own linear model with `lags` most recent returns.
    """

    def __init__(self, lags: int = 5):
        self.lags = lags
        self.models: list[LinearRegression] = []
        self.n_assets: int | None = None

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        x_np = to_numpy(x)
        y_np = to_numpy(y)
        returns = x_np[:, :, :, 0]
        n_assets = returns.shape[2]
        use_lags = min(self.lags, returns.shape[1])
        self.models = []
        self.n_assets = n_assets
        for asset_idx in range(n_assets):
            x_asset = returns[:, -use_lags:, asset_idx]
            y_asset = y_np[:, asset_idx]
            reg = LinearRegression()
            reg.fit(x_asset, y_asset)
            self.models.append(reg)

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.n_assets is None or not self.models:
            raise RuntimeError("LinearAutoregressiveModel must be fit before predict.")
        x_np = to_numpy(x)
        returns = x_np[:, :, :, 0]
        use_lags = min(self.lags, returns.shape[1])
        preds = np.zeros((returns.shape[0], self.n_assets), dtype=float)
        for asset_idx, reg in enumerate(self.models):
            x_asset = returns[:, -use_lags:, asset_idx]
            preds[:, asset_idx] = reg.predict(x_asset)
        return preds


class RidgeRegressionModel(BaselineModel):
    """Ridge on flattened context (all features)."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        self.model.fit(flatten_context(x), to_numpy(y))

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.model.predict(flatten_context(x))


class MLPBaselineModel(BaselineModel):
    """Shallow nonlinear baseline with train-only scaling."""

    def __init__(self, random_state: int = 0):
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        max_iter=300,
                        early_stopping=True,
                        n_iter_no_change=15,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        self.pipeline.fit(flatten_context(x), to_numpy(y))

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.pipeline.predict(flatten_context(x))


class RandomForestBaselineModel(BaselineModel):
    """Tree-based nonlinear baseline."""

    def __init__(self, random_state: int = 0):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        self.model.fit(flatten_context(x), to_numpy(y))

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.model.predict(flatten_context(x))
