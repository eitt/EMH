from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """
    Supervised dataset with horizon-consistent targets.

    For origin i:
    - Context x_i uses [i-L, ..., i-1].
    - Target y_i is cumulative return over [i, ..., i+H-1].
    """

    def __init__(self, returns: pd.DataFrame, mask: pd.DataFrame, amihud: pd.DataFrame, L: int, H: int):
        if L <= 0 or H <= 0:
            raise ValueError("L and H must be positive integers.")
        if not returns.index.is_monotonic_increasing:
            raise ValueError("returns index must be sorted by time.")
        if returns.index.has_duplicates:
            raise ValueError("returns index contains duplicate timestamps.")
        if not (returns.index.equals(mask.index) and returns.index.equals(amihud.index)):
            raise ValueError("Feature indices are not aligned.")
        if not (list(returns.columns) == list(mask.columns) == list(amihud.columns)):
            raise ValueError("Feature columns are not aligned.")

        self.L = L
        self.H = H
        self.dates = returns.index.to_numpy()

        self.data_X = np.stack([returns.values, mask.values, amihud.values], axis=-1)
        self.data_Y = returns.values

        # Inclusive upper bound for origin: i + H - 1 <= T-1  => i <= T-H
        max_origin = len(self.data_X) - H
        if max_origin < L:
            raise ValueError("Not enough observations for given (L, H).")
        self.origins = np.arange(L, max_origin + 1, dtype=int)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.origins[idx])
        x = self.data_X[i - self.L : i]
        y = self.data_Y[i : i + self.H].sum(axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _load_processed_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
    mask = pd.read_csv("data/processed/mask.csv", index_col=0, parse_dates=True)
    amihud = pd.read_csv("data/processed/amihud.csv", index_col=0, parse_dates=True)
    return returns, mask, amihud


def get_dataloaders(
    L: int,
    H: int,
    batch_size: int = 32,
    train_split: float = 0.8,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Standard single-split loaders with horizon gap.
    Used by trainer/xai convenience scripts.
    """
    if not (0.5 <= train_split < 0.95):
        raise ValueError("train_split must be in [0.5, 0.95).")

    returns, mask, amihud = _load_processed_frames()
    dataset = TimeSeriesDataset(returns, mask, amihud, L, H)

    n = len(dataset)
    train_size = int(n * train_split)
    gap = max(H - 1, 0)
    val_start = train_size + gap
    if val_start >= n:
        raise ValueError("Validation split empty after horizon gap.")

    # Leakage assertion: train target end must be strictly before first validation origin.
    train_last_origin = int(dataset.origins[train_size - 1])
    val_first_origin = int(dataset.origins[val_start])
    if not (train_last_origin + H - 1 < val_first_origin):
        raise AssertionError("Temporal leakage risk: overlapping train/validation target horizons.")

    train_ds = torch.utils.data.Subset(dataset, range(train_size))
    val_ds = torch.utils.data.Subset(dataset, range(val_start, n))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
