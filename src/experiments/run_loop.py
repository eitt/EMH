from __future__ import annotations

import logging
import os
import random
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from src.models.benchmarks.baselines import (
    ARIMAXModel,
    ARCHBaselineModel,
    GARCHBaselineModel,
    HistoricalMeanModel,
    LinearAutoregressiveModel,
    MLPBaselineModel,
    RandomForestBaselineModel,
    RandomWalkModel,
    RidgeRegressionModel,
    VARBaselineModel,
)
from src.models.diffusion.model import ConditionalDiffusionModel, DiffusionProcess
from src.stats.diebold_mariano import diebold_mariano_test
from src.stats.model_confidence_set import compute_model_confidence_set

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DiffusionTrainConfig:
    num_steps: int = 20
    hidden_dim: int = 64
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 12
    patience: int = 3


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
    mask = pd.read_csv("data/processed/mask.csv", index_col=0, parse_dates=True)
    amihud = pd.read_csv("data/processed/amihud.csv", index_col=0, parse_dates=True)

    if not returns.index.is_monotonic_increasing:
        raise ValueError("Returns index must be sorted in time.")
    if returns.index.has_duplicates:
        raise ValueError("Returns index contains duplicate timestamps.")
    if not (returns.index.equals(mask.index) and returns.index.equals(amihud.index)):
        raise ValueError("Processed feature indices are not aligned.")
    if not (list(returns.columns) == list(mask.columns) == list(amihud.columns)):
        raise ValueError("Processed feature columns are not aligned.")
    return returns, mask, amihud


def build_supervised_arrays(
    returns: pd.DataFrame,
    mask: pd.DataFrame,
    amihud: pd.DataFrame,
    L: int,
    H: int,
    feature_set: str = "full",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if feature_set not in {"full", "no_illiquidity", "returns_only"}:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    feature_stack = np.stack([returns.values, mask.values, amihud.values], axis=-1)
    if feature_set == "no_illiquidity":
        feature_stack = feature_stack[:, :, [0, 1]]
    elif feature_set == "returns_only":
        feature_stack = feature_stack[:, :, [0]]

    T = feature_stack.shape[0]
    max_origin = T - H
    if max_origin < L:
        raise ValueError("Not enough observations for selected (L, H).")

    origins = np.arange(L, max_origin + 1, dtype=int)
    X = np.stack([feature_stack[i - L : i] for i in origins], axis=0)
    y = np.stack([returns.values[i : i + H].sum(axis=0) for i in origins], axis=0)

    # Add regime conditioning: high volatility indicator
    # Compute std of returns over the L days for each asset
    returns_in_window = X[:, :, :, 0]  # [batch, L, N]
    vol_std = np.std(returns_in_window, axis=1, keepdims=True)  # [batch, 1, N]
    # High vol if std > median across all
    median_vol = np.median(vol_std)
    high_vol = (vol_std > median_vol).astype(float)  # [batch, 1, N]
    # Add as new feature
    high_vol_expanded = np.repeat(high_vol, L, axis=1)  # [batch, L, N]
    X = np.concatenate([X, high_vol_expanded[:, :, :, np.newaxis]], axis=-1)

    # Global regime label for each origin based on average volatility exposure.
    regime_label = (vol_std.mean(axis=(1, 2)) > median_vol).astype(int).reshape(-1)

    # Leakage guard at sample construction level.
    if origins[-1] + H - 1 >= T:
        raise AssertionError("Target horizon exceeds available timeline.")
    return X, y, origins, regime_label


def generate_expanding_windows(
    n_samples: int,
    H: int,
    n_windows: int = 5,
    min_train_frac: float = 0.55,
    val_frac: float = 0.12,
) -> list[dict[str, int]]:
    if n_samples < 200:
        n_windows = max(2, min(n_windows, 3))
    gap = max(H - 1, 0)
    min_train = max(int(n_samples * min_train_frac), 120)
    val_size = max(int(n_samples * val_frac), 60)
    start_min = min_train + gap
    start_max = n_samples - val_size
    if start_min >= start_max:
        raise ValueError("Insufficient samples for expanding-window evaluation.")

    starts = np.linspace(start_min, start_max, num=n_windows, dtype=int)
    starts = sorted(set(int(s) for s in starts))
    windows: list[dict[str, int]] = []
    for wid, val_start in enumerate(starts):
        train_end = val_start - gap
        val_end = min(val_start + val_size, n_samples)
        if train_end <= 0 or val_end <= val_start:
            continue
        windows.append(
            {
                "window_id": wid,
                "train_start": 0,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "gap": gap,
            }
        )
    if not windows:
        raise ValueError("No valid windows generated.")
    return windows


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    return float(np.mean(true_sign == pred_sign))


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    da = directional_accuracy(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "Directional_Accuracy": da}


def split_inner_train_valid(
    X_train: np.ndarray,
    y_train: np.ndarray,
    frac: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cut = int(len(X_train) * frac)
    cut = max(cut, min(len(X_train) - 20, 80))
    return X_train[:cut], y_train[:cut], X_train[cut:], y_train[cut:]


def fit_ridge_time_aware(X_train: np.ndarray, y_train: np.ndarray) -> RidgeRegressionModel:
    alphas = [0.1, 1.0, 5.0, 10.0]
    X_sub, y_sub, X_val, y_val = split_inner_train_valid(X_train, y_train)
    best_alpha = 1.0
    best_rmse = float("inf")
    for alpha in alphas:
        mdl = RidgeRegressionModel(alpha=alpha)
        mdl.fit(X_sub, y_sub)
        pred = mdl.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    model = RidgeRegressionModel(alpha=best_alpha)
    model.fit(X_train, y_train)
    return model


def fit_mlp_time_aware(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 0) -> MLPBaselineModel:
    hidden_options = [(16,), (32,), (32, 16)]
    alphas = [1e-4, 1e-3]
    X_sub, y_sub, X_val, y_val = split_inner_train_valid(X_train, y_train)
    best_rmse = float("inf")
    best_model = MLPBaselineModel(random_state=random_state)
    for hidden_layers in hidden_options:
        for alpha in alphas:
            mdl = MLPBaselineModel(
                hidden_layer_sizes=hidden_layers,
                alpha=alpha,
                max_iter=200,
                random_state=random_state,
            )
            mdl.fit(X_sub, y_sub)
            pred = mdl.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = mdl
    best_model.fit(X_train, y_train)
    return best_model


def fit_random_forest_time_aware(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 0) -> RandomForestBaselineModel:
    candidate_configs = [
        {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5},
    ]
    X_sub, y_sub, X_val, y_val = split_inner_train_valid(X_train, y_train)
    best_rmse = float("inf")
    best_model = RandomForestBaselineModel(random_state=random_state)
    for cfg in candidate_configs:
        mdl = RandomForestBaselineModel(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=random_state,
        )
        mdl.fit(X_sub, y_sub)
        pred = mdl.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = mdl
    best_model.fit(X_train, y_train)
    return best_model


def fit_arima_time_aware(
    X_train: np.ndarray,
    y_train: np.ndarray,
    orders: list[tuple[int, int, int]] | None = None,
) -> ARIMAXModel:
    if orders is None:
        orders = [(1, 0, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)]
    X_sub, y_sub, X_val, y_val = split_inner_train_valid(X_train, y_train)
    best_rmse = float("inf")
    best_order = orders[0]
    for order in orders:
        try:
            mdl = ARIMAXModel(order=order, use_exog=True)
            mdl.fit(X_sub, y_sub)
            pred = mdl.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_order = order
        except Exception:
            continue
    model = ARIMAXModel(order=best_order, use_exog=True)
    model.fit(X_train, y_train)
    return model


def fit_var_time_aware(X_train: np.ndarray, y_train: np.ndarray, lag_options: list[int] | None = None) -> VARBaselineModel:
    if lag_options is None:
        lag_options = [1, 2, 3]
    X_sub, y_sub, X_val, y_val = split_inner_train_valid(X_train, y_train)
    best_rmse = float("inf")
    best_lag = lag_options[0]
    for lags in lag_options:
        try:
            mdl = VARBaselineModel(lags=lags)
            mdl.fit(X_sub, y_sub)
            pred = mdl.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_lag = lags
        except Exception:
            continue
    model = VARBaselineModel(lags=best_lag)
    model.fit(X_train, y_train)
    return model


def tune_diffusion_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    candidate_cfgs: list[DiffusionTrainConfig],
    device: torch.device,
    predict_type: str = 'x0',
    schedule: str = 'cosine',
) -> tuple[ConditionalDiffusionModel, DiffusionProcess, dict[str, float]]:
    best_loss = float("inf")
    best_cfg = candidate_cfgs[0]
    for cfg in candidate_cfgs:
        try:
            _, _, meta = train_diffusion_model(
                X_train,
                y_train,
                seed=seed,
                cfg=cfg,
                device=device,
                predict_type=predict_type,
                schedule=schedule,
                full_train=False,
            )
            if meta["best_val_loss"] < best_loss:
                best_loss = meta["best_val_loss"]
                best_cfg = cfg
        except Exception:
            continue
    model, process, meta = train_diffusion_model(
        X_train,
        y_train,
        seed=seed,
        cfg=best_cfg,
        device=device,
        predict_type=predict_type,
        schedule=schedule,
        full_train=True,
    )
    meta["selected_hidden_dim"] = best_cfg.hidden_dim
    meta["selected_lr"] = best_cfg.lr
    meta["selected_num_steps"] = best_cfg.num_steps
    return model, process, meta


def make_torch_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_diffusion_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    cfg: DiffusionTrainConfig,
    device: torch.device,
    predict_type: str = 'noise',
    schedule: str = 'linear',
    full_train: bool = False,
) -> tuple[ConditionalDiffusionModel, DiffusionProcess, dict[str, float]]:
    set_global_seed(seed)
    if full_train:
        X_sub, y_sub = X_train, y_train
        X_val, y_val = None, None
    else:
        X_sub, y_sub, X_val, y_val = split_inner_train_valid(X_train, y_train)

    train_loader = make_torch_batches(X_sub, y_sub, cfg.batch_size, shuffle=True)
    val_loader = make_torch_batches(X_val, y_val, cfg.batch_size, shuffle=False) if X_val is not None else None

    target_dim = y_train.shape[1]
    context_dim = X_train.shape[2] * X_train.shape[3]
    context_seq_len = X_train.shape[1]
    model = ConditionalDiffusionModel(
        target_dim=target_dim,
        context_dim=context_dim,
        hidden_dim=cfg.hidden_dim,
        predict_type=predict_type,
        context_seq_len=context_seq_len,
    ).to(device)
    process = DiffusionProcess(num_steps=cfg.num_steps, schedule=schedule, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()

    best_state = None
    best_val = float("inf")
    patience_left = cfg.patience
    best_epoch = -1

    for epoch in range(cfg.max_epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device)
            y_b = y_b.to(device)
            context = X_b
            t_idx = torch.randint(0, cfg.num_steps, (X_b.shape[0],), device=device).long()
            y_noisy, noise, y_start = process.add_noise(y_b, t_idx)
            pred = model(y_noisy, t_idx.float().unsqueeze(1), context)
            
            if predict_type == 'noise':
                loss = criterion(pred, noise)
            elif predict_type == 'x0':
                pred_x0 = model.predict_x0(y_noisy, t_idx, pred, process.alphas_cumprod)
                loss = criterion(pred_x0, y_start)
            else:
                raise ValueError(f"Unknown predict_type: {predict_type}")
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        if val_loader is not None:
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v = X_v.to(device)
                    y_v = y_v.to(device)
                    ctx_v = X_v
                    t_idx = torch.randint(0, cfg.num_steps, (X_v.shape[0],), device=device).long()
                    y_noisy, noise, y_start = process.add_noise(y_v, t_idx)
                    pred = model(y_noisy, t_idx.float().unsqueeze(1), ctx_v)
                    
                    if predict_type == 'noise':
                        loss = criterion(pred, noise)
                    elif predict_type == 'x0':
                        pred_x0 = model.predict_x0(y_noisy, t_idx, pred, process.alphas_cumprod)
                        loss = criterion(pred_x0, y_start)
                    else:
                        raise ValueError(f"Unknown predict_type: {predict_type}")
                    
                    val_loss_sum += float(loss.item()) * X_v.shape[0]
                    val_count += X_v.shape[0]
            val_loss = val_loss_sum / max(val_count, 1)
        else:
            val_loss = float('inf')

        if val_loader is None:
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {"best_val_loss": float(best_val), "best_epoch": int(best_epoch)}
    return model, process, metadata


def predict_diffusion(
    model: ConditionalDiffusionModel,
    process: DiffusionProcess,
    X_eval: np.ndarray,
    batch_size: int,
    device: torch.device,
    n_samples: int = 1,
    agg: str = 'mean',
) -> np.ndarray:
    model.eval()
    preds = []
    loader = make_torch_batches(X_eval, np.zeros((len(X_eval), model.target_dim)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X_b, _ in loader:
            X_b = X_b.to(device)
            context = X_b
            samples = []
            for _ in range(n_samples):
                sample = process.sample(model, context, shape=(X_b.shape[0], model.target_dim))
                samples.append(sample)
            samples = torch.stack(samples, dim=0)  # [n_samples, batch, dim]
            if agg == 'mean':
                pred = samples.mean(dim=0)
            elif agg == 'median':
                pred = samples.median(dim=0).values
            else:
                raise ValueError(f"Unknown agg: {agg}")
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def permutation_importance_by_feature_channel(
    model: ConditionalDiffusionModel,
    process: DiffusionProcess,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    baseline_pred = predict_diffusion(model, process, X_val, batch_size=256, device=device, n_samples=10)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_pred)))

    channel_names = ["returns", "mask", "amihud", "regime"]
    channel_count = X_val.shape[3]
    output: dict[str, float] = {}
    for ch in range(channel_count):
        X_perm = X_val.copy()
        perm_idx = rng.permutation(len(X_perm))
        X_perm[:, :, :, ch] = X_perm[perm_idx, :, :, ch]
        pred_perm = predict_diffusion(model, process, X_perm, batch_size=256, device=device, n_samples=10)
        rmse_perm = float(np.sqrt(mean_squared_error(y_val, pred_perm)))
        output[channel_names[ch] if ch < len(channel_names) else f"feature_{ch}"] = rmse_perm - baseline_rmse
    return output


def compute_descriptive_diagnostics(returns: pd.DataFrame, reports_dir: Path) -> None:
    stats_rows = []
    diag_rows = []
    corr = returns.corr()

    # Price proxy from cumulative returns if raw close unavailable.
    raw_path = Path("data/raw/latest_raw.csv")
    close = None
    if raw_path.exists():
        raw = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True)
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy().reindex(returns.index).ffill()
    if close is None:
        close = np.exp(returns.cumsum())

    for ticker in returns.columns:
        s = returns[ticker].dropna()
        stats_rows.append(
            {
                "ticker": ticker,
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurtosis()),
                "acf1": float(s.autocorr(lag=1)),
            }
        )

        p = close[ticker].dropna()
        adf_price = adfuller(p, regression="c", autolag="AIC")[1] if len(p) > 50 else np.nan
        adf_ret = adfuller(s, regression="c", autolag="AIC")[1] if len(s) > 50 else np.nan
        lb_ret = acorr_ljungbox(s, lags=[10], return_df=True)["lb_pvalue"].iloc[0] if len(s) > 20 else np.nan
        lb_sq = (
            acorr_ljungbox((s**2), lags=[10], return_df=True)["lb_pvalue"].iloc[0] if len(s) > 20 else np.nan
        )
        diag_rows.append(
            {
                "ticker": ticker,
                "adf_price_p": float(adf_price),
                "adf_return_p": float(adf_ret),
                "lb10_return_p": float(lb_ret),
                "lb10_sqreturn_p": float(lb_sq),
            }
        )

    pd.DataFrame(stats_rows).to_csv(reports_dir / "summary_statistics.csv", index=False)
    corr_out = corr.reset_index().rename(columns={"index": "ticker"})
    corr_out.to_csv(reports_dir / "return_correlations.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(reports_dir / "diagnostic_tests.csv", index=False)


def run_main_robust_evaluation(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns, mask, amihud = load_processed_data()
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    compute_descriptive_diagnostics(returns, out_dir)

    configs = [(10, 1), (10, 5), (21, 1), (21, 5), (42, 1), (42, 5)]
    diffusion_cfg = DiffusionTrainConfig()

    per_window_rows: list[dict[str, object]] = []
    per_asset_rows: list[dict[str, object]] = []
    regime_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    dm_store: dict[tuple[int, int, str], dict[str, list[np.ndarray]]] = {}
    diffusion_training_rows: list[dict[str, object]] = []

    asset_names = list(returns.columns)

    for L, H in configs:
        logger.info("Robust evaluation: L=%s, H=%s", L, H)
        X, y, origins, regimes = build_supervised_arrays(returns, mask, amihud, L=L, H=H, feature_set="full")
        windows = generate_expanding_windows(n_samples=len(X), H=H, n_windows=5)

        for w in windows:
            w_id = w["window_id"]
            tr0, tr1 = w["train_start"], w["train_end"]
            va0, va1 = w["val_start"], w["val_end"]

            X_train, y_train = X[tr0:tr1], y[tr0:tr1]
            X_val, y_val = X[va0:va1], y[va0:va1]
            if len(X_train) < 100 or len(X_val) < 20:
                continue

            train_last_origin = int(origins[tr1 - 1])
            val_first_origin = int(origins[va0])
            no_overlap = bool(train_last_origin + H - 1 < val_first_origin)
            leakage_rows.append(
                {
                    "L": L,
                    "H": H,
                    "window_id": w_id,
                    "train_last_origin": train_last_origin,
                    "val_first_origin": val_first_origin,
                    "horizon": H,
                    "no_target_overlap": no_overlap,
                }
            )
            if not no_overlap:
                raise AssertionError(f"Leakage guard failed for L={L}, H={H}, window={w_id}")

            model_map = {
                "Random Walk": RandomWalkModel(),
                "Historical Mean": HistoricalMeanModel(),
                "AR Linear": LinearAutoregressiveModel(lags=min(5, L)),
                "ARIMA": ARIMAXModel(),
                "ARCH": ARCHBaselineModel(),
                "GARCH": GARCHBaselineModel(),
                "VAR": VARBaselineModel(),
                "Ridge": RidgeRegressionModel(),
                "MLP": MLPBaselineModel(random_state=1000 + 17 * w_id + L + H),
                "Random Forest": RandomForestBaselineModel(random_state=2000 + 19 * w_id + L + H),
            }

            # Fit all benchmark models on the current training window before forecasting.
            for model in model_map.values():
                model.fit(X_train, y_train)

            pred_map: dict[str, np.ndarray] = {}
            for name, model in model_map.items():
                pred_map[name] = model.predict(X_val)

            candidate_cfgs = [
                DiffusionTrainConfig(hidden_dim=64, lr=1e-3, num_steps=20, batch_size=128, max_epochs=12, patience=3),
                DiffusionTrainConfig(hidden_dim=64, lr=5e-4, num_steps=20, batch_size=128, max_epochs=12, patience=3),
                DiffusionTrainConfig(hidden_dim=128, lr=1e-3, num_steps=20, batch_size=128, max_epochs=12, patience=3),
            ]
            diff_model, diff_proc, diff_meta = tune_diffusion_model(
                X_train,
                y_train,
                seed=42 + w_id,
                candidate_cfgs=candidate_cfgs,
                device=device,
                predict_type='x0',
                schedule='cosine',
            )
            pred_map["Diffusion"] = predict_diffusion(diff_model, diff_proc, X_val, batch_size=256, device=device, n_samples=10)
            diffusion_training_rows.append(
                {
                    "L": L,
                    "H": H,
                    "window_id": w_id,
                    "best_epoch": diff_meta["best_epoch"],
                    "best_val_loss": diff_meta.get("best_val_loss", float("nan")),
                    "selected_hidden_dim": diff_meta.get("selected_hidden_dim", None),
                    "selected_lr": diff_meta.get("selected_lr", None),
                    "selected_num_steps": diff_meta.get("selected_num_steps", None),
                }
            )

            origin_date_start = str(returns.index[int(origins[va0])].date())
            origin_date_end = str(returns.index[int(origins[va1 - 1])].date())
            window_regimes = regimes[va0:va1]
            for model_name, pred in pred_map.items():
                metrics = evaluate_metrics(y_val, pred)
                per_window_rows.append(
                    {
                        "L": L,
                        "H": H,
                        "window_id": w_id,
                        "model": model_name,
                        "val_origin_start": origin_date_start,
                        "val_origin_end": origin_date_end,
                        "n_train": int(len(X_train)),
                        "n_val": int(len(X_val)),
                        "RMSE": metrics["RMSE"],
                        "MAE": metrics["MAE"],
                        "Directional_Accuracy": metrics["Directional_Accuracy"],
                    }
                )

                for asset_idx, ticker in enumerate(asset_names):
                    asset_metrics = evaluate_metrics(
                        y_val[:, asset_idx : asset_idx + 1], pred[:, asset_idx : asset_idx + 1]
                    )
                    per_asset_rows.append(
                        {
                            "L": L,
                            "H": H,
                            "window_id": w_id,
                            "model": model_name,
                            "ticker": ticker,
                            "val_origin_start": origin_date_start,
                            "val_origin_end": origin_date_end,
                            "RMSE": asset_metrics["RMSE"],
                            "MAE": asset_metrics["MAE"],
                            "Directional_Accuracy": asset_metrics["Directional_Accuracy"],
                        }
                    )

                for regime_value in np.unique(window_regimes):
                    regime_mask = window_regimes == regime_value
                    if regime_mask.sum() == 0:
                        continue
                    y_reg = y_val[regime_mask]
                    pred_reg = pred[regime_mask]
                    regime_metrics = evaluate_metrics(y_reg, pred_reg)
                    regime_rows.append(
                        {
                            "L": L,
                            "H": H,
                            "window_id": w_id,
                            "model": model_name,
                            "regime": "high_vol" if int(regime_value) == 1 else "low_vol",
                            "n_eval": int(regime_mask.sum()),
                            "RMSE": regime_metrics["RMSE"],
                            "MAE": regime_metrics["MAE"],
                            "Directional_Accuracy": regime_metrics["Directional_Accuracy"],
                        }
                    )

                if model_name != "Random Walk":
                    key = (L, H, model_name)
                    dm_store.setdefault(key, {"y": [], "rw": [], "m": []})
                    dm_store[key]["y"].append(y_val.ravel())
                    dm_store[key]["rw"].append(pred_map["Random Walk"].ravel())
                    dm_store[key]["m"].append(pred.ravel())

    per_window_df = pd.DataFrame(per_window_rows)
    if per_window_df.empty:
        raise RuntimeError("No per-window results generated.")

    agg = (
        per_window_df.groupby(["L", "H", "model"], as_index=False)
        .agg(
            RMSE=("RMSE", "mean"),
            MAE=("MAE", "mean"),
            Directional_Accuracy=("Directional_Accuracy", "mean"),
            RMSE_STD=("RMSE", "std"),
            N_Windows=("window_id", "nunique"),
            N_Val_Total=("n_val", "sum"),
        )
        .sort_values(["L", "H", "RMSE"])
        .rename(columns={"model": "Model"})
    )

    dm_rows = []
    for (L, H, model_name), values in sorted(dm_store.items()):
        y_true = np.concatenate(values["y"])
        y_rw = np.concatenate(values["rw"])
        y_m = np.concatenate(values["m"])
        dm_stat, p_val = diebold_mariano_test(y_true, y_rw, y_m, h=H, loss="mse")
        dm_rows.append(
            {
                "L": L,
                "H": H,
                "Model": model_name,
                "DM_Stat": float(dm_stat),
                "P_Value": float(p_val),
                "significant_5pct": bool(p_val < 0.05),
            }
        )
    dm_df = pd.DataFrame(dm_rows)

    experiment_df = agg.merge(dm_df, on=["L", "H", "Model"], how="left")
    experiment_df.loc[experiment_df["Model"] == "Random Walk", ["DM_Stat", "P_Value", "significant_5pct"]] = [
        0.0,
        1.0,
        False,
    ]
    experiment_df = experiment_df.sort_values(["L", "H", "RMSE"]).reset_index(drop=True)

    rmse_pivot = experiment_df.pivot(index=["L", "H"], columns="Model", values="RMSE").reset_index()
    leakage_df = pd.DataFrame(leakage_rows)
    diffusion_train_df = pd.DataFrame(diffusion_training_rows)

    mcs_rows: list[dict[str, object]] = []
    for (l_val, h_val), group in per_window_df.groupby(["L", "H"]):
        loss_pivot = group.pivot(index="window_id", columns="model", values="RMSE")
        if loss_pivot.shape[1] < 2:
            continue
        mcs_result = compute_model_confidence_set(
            loss_pivot.values,
            list(loss_pivot.columns),
            alpha=0.05,
            n_bootstrap=300,
            seed=42,
        )
        mcs_rows.append(
            {
                "L": int(l_val),
                "H": int(h_val),
                "models_in_mcs": ",".join(mcs_result["kept_models"]),
                "models_excluded": ",".join(mcs_result["excluded_models"]),
                "mcs_level": float(mcs_result["mcs_level"]),
                "p_values": ",".join(str(mcs_result["p_values"][name]) for name in loss_pivot.columns),
            }
        )

    mcs_df = pd.DataFrame(mcs_rows)

    # Model rank stability by window.
    rank_df = per_window_df.copy()
    rank_df["rmse_rank"] = rank_df.groupby(["L", "H", "window_id"])["RMSE"].rank(method="min")
    rank_stability = (
        rank_df.groupby("model", as_index=False)
        .agg(
            mean_rank=("rmse_rank", "mean"),
            wins=("rmse_rank", lambda s: int(np.sum(s == 1))),
            windows=("rmse_rank", "count"),
        )
        .sort_values("mean_rank")
    )

    experiment_df.to_csv(out_dir / "experiment_results.csv", index=False)
    per_window_df.to_csv(out_dir / "per_window_losses.csv", index=False)
    pd.DataFrame(per_asset_rows).to_csv(out_dir / "per_asset_losses.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(out_dir / "regime_performance.csv", index=False)
    mcs_df.to_csv(out_dir / "mcs_results.csv", index=False)
    dm_df.to_csv(out_dir / "dm_tests.csv", index=False)
    rmse_pivot.to_csv(out_dir / "rmse_pivot.csv", index=False)
    leakage_df.to_csv(out_dir / "leakage_checks.csv", index=False)
    diffusion_train_df.to_csv(out_dir / "diffusion_training_stability.csv", index=False)
    rank_stability.to_csv(out_dir / "model_rank_stability.csv", index=False)

    return experiment_df, per_window_df


def run_diffusion_sensitivity(device: torch.device) -> None:
    returns, mask, amihud = load_processed_data()
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2]
    feature_sets = ["full", "no_illiquidity", "returns_only"]
    configs = [(21, 1), (21, 5)]
    cfg = DiffusionTrainConfig(max_epochs=10, patience=2, num_steps=20, hidden_dim=64)

    sensitivity_rows = []
    xai_rows = []
    xai_vectors: dict[tuple[int, int], dict[int, np.ndarray]] = {}

    for L, H in configs:
        for feature_set in feature_sets:
            X, y, origins, _ = build_supervised_arrays(returns, mask, amihud, L=L, H=H, feature_set=feature_set)
            windows = generate_expanding_windows(len(X), H=H, n_windows=2)
            for seed in seeds:
                rmse_list = []
                mae_list = []
                for w in windows:
                    tr0, tr1 = w["train_start"], w["train_end"]
                    va0, va1 = w["val_start"], w["val_end"]
                    X_train, y_train = X[tr0:tr1], y[tr0:tr1]
                    X_val, y_val = X[va0:va1], y[va0:va1]

                    train_last_origin = int(origins[tr1 - 1])
                    val_first_origin = int(origins[va0])
                    if not (train_last_origin + H - 1 < val_first_origin):
                        raise AssertionError("Leakage guard failed in diffusion sensitivity.")

                    model, process, _ = train_diffusion_model(X_train, y_train, seed=seed, cfg=cfg, device=device, predict_type='x0', schedule='cosine')
                    pred = predict_diffusion(model, process, X_val, batch_size=256, device=device, n_samples=10)
                    rmse_list.append(float(np.sqrt(mean_squared_error(y_val, pred))))
                    mae_list.append(float(mean_absolute_error(y_val, pred)))

                    # XAI-style model reliance for full feature set only.
                    if feature_set == "full" and w["window_id"] == 0:
                        imp = permutation_importance_by_feature_channel(model, process, X_val, y_val, seed=seed, device=device)
                        vec = np.array([imp.get("returns", 0.0), imp.get("mask", 0.0), imp.get("amihud", 0.0)])
                        z = (vec - vec.mean()) / (vec.std() + 1e-8)
                        max_z = float(np.max(z))
                        xai_rows.append(
                            {
                                "L": L,
                                "H": H,
                                "seed": seed,
                                "importance_returns": vec[0],
                                "importance_mask": vec[1],
                                "importance_amihud": vec[2],
                                "max_z_score": max_z,
                                "reject_threshold_2p5": bool(max_z > 2.5),
                            }
                        )
                        xai_vectors.setdefault((L, H), {})[seed] = vec

                sensitivity_rows.append(
                    {
                        "L": L,
                        "H": H,
                        "feature_set": feature_set,
                        "seed": seed,
                        "rmse_mean": float(np.mean(rmse_list)),
                        "mae_mean": float(np.mean(mae_list)),
                        "windows": len(rmse_list),
                    }
                )

    sens_df = pd.DataFrame(sensitivity_rows)
    if sens_df.empty:
        return
    sens_df.to_csv(out_dir / "diffusion_seed_sensitivity.csv", index=False)
    ablation = (
        sens_df.groupby(["L", "H", "feature_set"], as_index=False)
        .agg(rmse_mean=("rmse_mean", "mean"), mae_mean=("mae_mean", "mean"), rmse_std=("rmse_mean", "std"))
        .sort_values(["L", "H", "rmse_mean"])
    )
    ablation.to_csv(out_dir / "diffusion_ablation.csv", index=False)

    xai_df = pd.DataFrame(xai_rows)
    if not xai_df.empty:
        xai_df.to_csv(out_dir / "xai_seed_sensitivity.csv", index=False)

        stability_rows = []
        for (L, H), seed_vecs in xai_vectors.items():
            for s1, s2 in combinations(sorted(seed_vecs), 2):
                rho = float(spearmanr(seed_vecs[s1], seed_vecs[s2]).correlation)
                stability_rows.append({"L": L, "H": H, "seed_a": s1, "seed_b": s2, "spearman_rho": rho})
        pd.DataFrame(stability_rows).to_csv(out_dir / "xai_stability.csv", index=False)


def build_robustness_summary() -> None:
    out_dir = Path("reports/tables")
    per_window_path = out_dir / "per_window_losses.csv"
    leakage_path = out_dir / "leakage_checks.csv"
    mcs_path = out_dir / "mcs_results.csv"
    if not (per_window_path.exists() and leakage_path.exists()):
        return
    per_window = pd.read_csv(per_window_path)
    leakage = pd.read_csv(leakage_path)
    summary = [
        {"item": "Evaluation windows", "value": int(per_window["window_id"].nunique())},
        {"item": "Total model-window evaluations", "value": int(len(per_window))},
        {"item": "Leakage checks passed", "value": bool(leakage["no_target_overlap"].all())},
        {"item": "Distinct benchmark models", "value": int(per_window["model"].nunique() - 1)},
    ]
    if mcs_path.exists():
        mcs = pd.read_csv(mcs_path)
        summary.append({"item": "MCS config rows", "value": int(len(mcs))})
    pd.DataFrame(summary).to_csv(out_dir / "robustness_summary.csv", index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    exp_df, per_window_df = run_main_robust_evaluation(device=device)
    run_diffusion_sensitivity(device=device)
    build_robustness_summary()

    logger.info("Saved robust experiment outputs to reports/tables/")
    logger.info("Top results by RMSE:\n%s", exp_df.sort_values(["L", "H", "RMSE"]).head(20))
