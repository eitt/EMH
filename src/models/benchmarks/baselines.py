from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR


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


def _reconstruct_series(series_windows: np.ndarray) -> np.ndarray:
    reconstructed = [series_windows[0].astype(float)]
    for row in series_windows[1:]:
        last_step = row[-1:]
        reconstructed.append(last_step.astype(float))
    return np.concatenate(reconstructed, axis=0)


class ARIMAXModel(BaselineModel):
    """
    Univariate ARIMA/ARIMAX baseline per asset.

    This benchmark fits one ARIMA model per asset using the historical return series
    reconstructed from overlapping training windows. Exogenous variables can include
    lagged illiquidity if requested.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0), use_exog: bool = True):
        self.order = order
        self.use_exog = use_exog
        self.models: list[ARIMA] = []
        self.n_assets: int | None = None
        self.horizon: int | None = None

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        x_np = to_numpy(x)
        self.horizon = to_numpy(y).shape[1]
        returns = x_np[:, :, :, 0]
        amihud = x_np[:, :, :, 2] if x_np.shape[3] > 2 else None
        n_assets = returns.shape[2]
        self.models = []
        self.n_assets = n_assets
        for asset_idx in range(n_assets):
            endog = _reconstruct_series(returns[:, :, asset_idx])
            exog = None
            if self.use_exog and amihud is not None:
                exog = _reconstruct_series(amihud[:, :, asset_idx]).reshape(-1, 1)
            model = ARIMA(endog, order=self.order, exog=exog)
            fitted = model.fit(method_kwargs={"warn_convergence": False})
            self.models.append(fitted)

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.n_assets is None or not self.models or self.horizon is None:
            raise RuntimeError("ARIMAXModel must be fit before predict.")
        x_np = to_numpy(x)
        n_samples = x_np.shape[0]
        preds = np.zeros((n_samples, self.n_assets), dtype=float)
        for asset_idx, fitted in enumerate(self.models):
            exog_last = None
            if self.use_exog and x_np.shape[3] > 2:
                last_amihud = x_np[:, -1, asset_idx, 2].reshape(-1, 1)
                exog_last = np.repeat(last_amihud, self.horizon, axis=0).reshape(n_samples, self.horizon, 1)
            for i in range(n_samples):
                exog_future = exog_last[i] if exog_last is not None else None
                forecast = fitted.forecast(steps=self.horizon, exog=exog_future)
                preds[i, asset_idx] = np.sum(np.asarray(forecast, dtype=float))
        return preds


class VARBaselineModel(BaselineModel):
    """
    VAR benchmark using the multivariate return panel.

    The model is built from the reconstructed historical return series and produces
    cumulative horizon forecasts by summing multi-step VAR predictions.
    """

    def __init__(self, lags: int = 1):
        self.lags = lags
        self.results = None
        self.n_assets: int | None = None
        self.horizon: int | None = None

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        x_np = to_numpy(x)
        self.horizon = to_numpy(y).shape[1]
        returns = x_np[:, :, :, 0]
        self.n_assets = returns.shape[2]
        series = returns[0].astype(float)
        for window in returns[1:]:
            series = np.vstack([series, window[-1:].astype(float)])
        model = VAR(series)
        self.results = model.fit(maxlags=self.lags, ic=None)

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.results is None or self.n_assets is None or self.horizon is None:
            raise RuntimeError("VARBaselineModel must be fit before predict.")
        x_np = to_numpy(x)
        n_samples = x_np.shape[0]
        preds = np.zeros((n_samples, self.n_assets), dtype=float)
        for i in range(n_samples):
            last_obs = x_np[i, -self.results.k_ar :, :, 0].astype(float)
            forecast = self.results.forecast(last_obs, steps=self.horizon)
            preds[i] = np.sum(forecast, axis=0)
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
