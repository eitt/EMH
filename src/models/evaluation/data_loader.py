import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    """
    Dataset for supervised prediction.
    X: [batch, features, lookback]
    Y: [batch, targets, horizon]
    """
    def __init__(self, returns, mask, amihud, L, H):
        self.L = L
        self.H = H
        
        # Combine features: [T, N, 3] -> (Returns, Mask, Amihud)
        # We'll align them
        self.data_X = np.stack([returns.values, mask.values, amihud.values], axis=-1)
        self.data_Y = returns.values
        
        # Determine valid start/end points
        # Need L historical points and H future points
        self.valid_indices = range(L, len(self.data_X) - H)
        
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        
        # Context X: [L, N, 3]
        x = self.data_X[i-self.L : i]
        # Target Y: Sum of returns over H: [N]
        # We forecast cumulative returns over the horizon H
        y = self.data_Y[i : i+self.H].sum(axis=0)
        
        return (torch.tensor(x, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.float32))

def get_dataloaders(L, H, batch_size=32, train_split=0.8):
    """Loads processed data and returns Train/Val split dataloaders."""
    returns = pd.read_csv('data/processed/returns.csv', index_col=0)
    mask = pd.read_csv('data/processed/mask.csv', index_col=0)
    amihud = pd.read_csv('data/processed/amihud.csv', index_col=0)
    
    dataset = TimeSeriesDataset(returns, mask, amihud, L, H)
    
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    # Simple split (CAUSAL: no shuffling before split)
    # We take the first train_size indices for training
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
