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
from statsmodels.regression.linear_model import OLS


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


class ARCHBaselineModel(BaselineModel):
    """
    Simple ARCH(1) econometric benchmark.

    The model fits an AR(1) mean equation for each asset and an ARCH(1)
    model for the squared residuals. The predictive mean is used as a classical
    econometric comparator in cumulative return forecasting.
    """

    def __init__(self, use_mean: bool = True):
        self.use_mean = use_mean
        self.ar_models: list[OLS] = []
        self.omega: list[float] = []
        self.alpha: list[float] = []
        self.n_assets: int | None = None
        self.horizon: int | None = None

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        x_np = to_numpy(x)
        self.horizon = to_numpy(y).shape[1]
        returns = x_np[:, :, :, 0]
        self.n_assets = returns.shape[2]
        self.ar_models = []
        self.omega = []
        self.alpha = []

        for asset_idx in range(self.n_assets):
            series = _reconstruct_series(returns[:, :, asset_idx]).astype(float)
            if len(series) < 10:
                raise RuntimeError("Not enough observations to fit ARCH model.")

            y_mean = series[1:]
            X_mean = np.column_stack([np.ones(len(y_mean)), series[:-1]])
            ar_model = OLS(y_mean, X_mean).fit()
            residuals = y_mean - ar_model.predict(X_mean)
            squared = residuals ** 2
            squared_lag = np.concatenate([[0.0], squared[:-1]])
            X_var = np.column_stack([np.ones(len(squared)), squared_lag])
            params = OLS(squared, X_var).fit().params
            self.ar_models.append(ar_model)
            self.omega.append(float(params[0]))
            self.alpha.append(float(params[1]))

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.n_assets is None or self.horizon is None:
            raise RuntimeError("ARCHBaselineModel must be fit before predict.")
        x_np = to_numpy(x)
        n_samples = x_np.shape[0]
        preds = np.zeros((n_samples, self.n_assets), dtype=float)

        for asset_idx, ar_model in enumerate(self.ar_models):
            last_return = x_np[:, -1, asset_idx, 0]
            params = ar_model.params
            mean_forecast = params[0] + params[1] * last_return
            preds[:, asset_idx] = mean_forecast
        return preds


class GARCHBaselineModel(BaselineModel):
    """
    Simple GARCH(1,1)-style econometric benchmark.

    The model fits an AR(1) mean equation for each asset and a GARCH(1,1)
    recurrence for conditional variance. The predicted mean is extrapolated
    over the horizon to produce cumulative return forecasts.
    """

    def __init__(self, use_mean: bool = True):
        self.use_mean = use_mean
        self.ar_models: list[OLS] = []
        self.omega: list[float] = []
        self.alpha: list[float] = []
        self.beta: list[float] = []
        self.last_cond_var: list[float] = []
        self.n_assets: int | None = None
        self.horizon: int | None = None

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        x_np = to_numpy(x)
        self.horizon = to_numpy(y).shape[1]
        returns = x_np[:, :, :, 0]
        self.n_assets = returns.shape[2]
        self.ar_models = []
        self.omega = []
        self.alpha = []
        self.beta = []
        self.last_cond_var = []

        for asset_idx in range(self.n_assets):
            series = _reconstruct_series(returns[:, :, asset_idx]).astype(float)
            if len(series) < 10:
                raise RuntimeError("Not enough observations to fit GARCH model.")

            y_mean = series[1:]
            X_mean = np.column_stack([np.ones(len(y_mean)), series[:-1]])
            ar_model = OLS(y_mean, X_mean).fit()
            residuals = y_mean - ar_model.predict(X_mean)
            squared = residuals ** 2
            var_lag = np.concatenate([[np.var(squared)], squared[:-1]])
            y_var = squared[1:]
            X_var = np.column_stack([np.ones(len(y_var)), squared[:-1], var_lag[:-1]])
            garch_fit = OLS(y_var, X_var).fit()
            params = garch_fit.params

            self.ar_models.append(ar_model)
            self.omega.append(float(params[0]))
            self.alpha.append(float(params[1]))
            self.beta.append(float(params[2]))
            self.last_cond_var.append(float(max(var_lag[-1], 1e-8)))

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.n_assets is None or self.horizon is None:
            raise RuntimeError("GARCHBaselineModel must be fit before predict.")
        x_np = to_numpy(x)
        n_samples = x_np.shape[0]
        preds = np.zeros((n_samples, self.n_assets), dtype=float)

        for asset_idx, ar_model in enumerate(self.ar_models):
            last_return = x_np[:, -1, asset_idx, 0]
            params = ar_model.params
            mean_forecast = params[0] + params[1] * last_return
            preds[:, asset_idx] = mean_forecast * float(self.horizon)
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

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        alpha: float = 0.0001,
        max_iter: int = 300,
        random_state: int = 0,
    ):
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        alpha=alpha,
                        activation="relu",
                        max_iter=max_iter,
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

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_leaf: int = 5,
        random_state: int = 0,
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        self.model.fit(flatten_context(x), to_numpy(y))

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        return self.model.predict(flatten_context(x))
