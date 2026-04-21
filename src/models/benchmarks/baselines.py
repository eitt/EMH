import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

class BaselineModel:
    def predict(self, X):
        raise NotImplementedError

class RandomWalkModel(BaselineModel):
    """Expects returns. Random Walk forecast for returns is 0."""
    def predict(self, X):
        # X: [batch, L, N, features]
        batch_size = X.shape[0]
        num_assets = X.shape[2]
        return torch.zeros(batch_size, num_assets)

class HistoricalMeanModel(BaselineModel):
    """Predicts the mean of returns observed in the lookback window."""
    def predict(self, X):
        # X: [batch, L, N, features]
        # features[0] is returns
        returns = X[:, :, :, 0]
        return returns.mean(dim=1)

class RidgeRegressionModel(BaselineModel):
    """Ridge regression trained on flattened context."""
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X, y):
        # Flatten X: [batch, L*N*features]
        X_flat = X.view(X.shape[0], -1).numpy()
        y_np = y.numpy()
        self.model.fit(X_flat, y_np)
        
    def predict(self, X):
        X_flat = X.view(X.shape[0], -1).numpy()
        preds = self.model.predict(X_flat)
        return torch.tensor(preds, dtype=torch.float32)

def evaluate_baseline(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in dataloader:
            preds = model.predict(X)
            all_preds.append(preds)
            all_targets.append(y)
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)
    
    return {"RMSE": rmse, "R2": r2}

if __name__ == "__main__":
    from src.models.evaluation.data_loader import get_dataloaders
    
    L, H = 21, 5
    train_loader, val_loader = get_dataloaders(L, H)
    
    models = {
        "Random Walk": RandomWalkModel(),
        "Historical Mean": HistoricalMeanModel(),
        "Ridge": RidgeRegressionModel()
    }
    
    # Train Ridge
    X_train, y_train = next(iter(DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset))))
    models["Ridge"].fit(X_train, y_train)
    
    for name, model in models.items():
        metrics = evaluate_baseline(model, val_loader)
        print(f"Model: {name} | Metrics: {metrics}")
