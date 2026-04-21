import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_data(file_path):
    """Loads raw yfinance CSV with multi-index columns."""
    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    return df

def preprocess_pipeline(raw_file, output_dir='data/processed'):
    """
    Main preprocessing pipeline for EMH analysis.
    Ensures causality and consistency.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_raw_data(raw_file)
    
    # Separate Close and Volume
    # Note: Depending on yfinance version, the structure might be (Ticker, Metric) or (Metric, Ticker)
    # We'll normalize to (Metric, Ticker) if needed, but usually it's (Metric, Ticker) if multiple tickers are passed.
    
    close = df['Close']
    volume = df['Volume']
    
    # 1. Forward fill and create mask
    mask = ((~close.isna()) & (volume > 0)).astype(float)
    close_filled = close.ffill()
    
    # 2. Log Returns (Causal: pt / pt-1)
    returns = np.log(close_filled / close_filled.shift(1)).fillna(0) * mask
    
    # 3. Amihud Illiquidity (|R| / (Vol * Price))
    # We use log returns absolute and dollar volume
    dollar_volume = volume * close_filled + 1e-8
    amihud = (returns.abs() / dollar_volume).fillna(0)
    
    # 4. Standardize Amihud (Causal: using rolling window)
    window = 60
    # We use shift(1) to avoid leakage of the current point into its own standardization parameters
    # Actually, a rolling mean up to t can be used if we only use information available at t.
    # But for PREDICTION tasks, we must be careful.
    ami_roll_mean = amihud.rolling(window, min_periods=window).mean()
    ami_roll_std = amihud.rolling(window, min_periods=window).std() + 1e-8
    amihud_std = (amihud - ami_roll_mean) / ami_roll_std
    
    # Combine into a single processed dataframe or separate files
    # We'll save a combined feather or paracquet file for efficiency, but CSV for transparency here
    processed_data = {
        'returns': returns,
        'mask': mask,
        'amihud': amihud_std.fillna(0)
    }
    
    for key, data in processed_data.items():
        out_path = os.path.join(output_dir, f'{key}.csv')
        data.to_csv(out_path)
        logger.info(f"Saved {key} to {out_path}")
    
    logger.info("Preprocessing complete.")
    return processed_data

if __name__ == "__main__":
    latest_raw = 'data/raw/latest_raw.csv'
    if os.path.exists(latest_raw):
        preprocess_pipeline(latest_raw)
    else:
        logger.error(f"Raw data not found at {latest_raw}")
