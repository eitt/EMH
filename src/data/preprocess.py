from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """Load yfinance CSV with multi-index columns."""
    return pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)


def _assert_temporal_integrity(df: pd.DataFrame) -> None:
    if not df.index.is_monotonic_increasing:
        raise ValueError("Raw data index is not sorted in increasing time order.")
    if df.index.has_duplicates:
        raise ValueError("Raw data index has duplicate timestamps.")


def _extract_close_volume(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # yfinance layout is usually [PriceType, Ticker].
    if "Close" in df.columns.get_level_values(0) and "Volume" in df.columns.get_level_values(0):
        close = df["Close"].copy()
        volume = df["Volume"].copy()
        return close, volume
    raise KeyError("Expected multi-index columns with top-level 'Close' and 'Volume'.")


def preprocess_pipeline(raw_file: str | Path, output_dir: str | Path = "data/processed") -> dict[str, pd.DataFrame]:
    """
    Main preprocessing pipeline with explicit temporal-leakage safeguards.

    Features:
    - returns: r_t = log(P_t/P_{t-1}) * mask_t
    - mask: 1 if price observed and volume > 0
    - amihud: strictly lagged z-score of illiquidity proxy
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_file)
    _assert_temporal_integrity(df)
    close, volume = _extract_close_volume(df)

    if not close.index.equals(volume.index):
        raise ValueError("Close and volume indices are not aligned.")
    if list(close.columns) != list(volume.columns):
        raise ValueError("Close and volume tickers are not aligned.")

    # Tradability mask.
    mask = ((~close.isna()) & (volume > 0)).astype(float)

    # Price forward-fill keeps causality (uses past values only).
    close_filled = close.ffill()

    # Log returns.
    returns = np.log(close_filled / close_filled.shift(1)).fillna(0.0) * mask

    # Amihud: |r_t| / (P_t * V_t)
    dollar_volume = close_filled * volume + 1e-8
    amihud_raw = (returns.abs() / dollar_volume).fillna(0.0)

    # Strictly lagged standardization:
    # stats at time t are computed with data up to t-1.
    window = 60
    roll_mean = amihud_raw.rolling(window=window, min_periods=window).mean().shift(1)
    roll_std = amihud_raw.rolling(window=window, min_periods=window).std().shift(1) + 1e-8
    amihud_std = ((amihud_raw - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    processed_data = {"returns": returns, "mask": mask, "amihud": amihud_std}

    for key, frame in processed_data.items():
        out_csv = output_path / f"{key}.csv"
        frame.to_csv(out_csv)
        logger.info("Saved %s to %s", key, out_csv)

    # Minimal metadata for manuscript and reproducibility.
    metadata = {
        "start_date": str(returns.index.min().date()),
        "end_date": str(returns.index.max().date()),
        "n_obs": int(returns.shape[0]),
        "n_assets": int(returns.shape[1]),
        "assets": [str(c) for c in returns.columns],
        "amihud_standardization_window": window,
        "amihud_standardization_strictly_lagged": True,
    }
    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata to %s", metadata_path)

    logger.info("Preprocessing complete.")
    return processed_data


if __name__ == "__main__":
    latest_raw = Path("data/raw/latest_raw.csv")
    if latest_raw.exists():
        preprocess_pipeline(latest_raw)
    else:
        logger.error("Raw data not found at %s", latest_raw)
