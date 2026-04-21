import pandas as pd
import numpy as np
import torch
import os
import logging
from src.models.evaluation.data_loader import get_dataloaders
from src.models.benchmarks.baselines import RandomWalkModel, HistoricalMeanModel, RidgeRegressionModel
from src.models.diffusion.model import ConditionalDiffusionModel, DiffusionProcess
from src.stats.diebold_mariano import diebold_mariano_test
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_models(L, H, device='cpu'):
    logger.info(f"Running experiment for L={L}, H={H}")
    
    train_loader, val_loader = get_dataloaders(L, H)
    target_dim = 4
    context_dim = L * 12
    
    # Init Baselines
    rw = RandomWalkModel()
    hm = HistoricalMeanModel()
    ridge = RidgeRegressionModel()
    
    # Train Ridge
    X_train, y_train = next(iter(torch.utils.data.DataLoader(train_loader.dataset, batch_size=len(train_loader.dataset))))
    ridge.fit(X_train, y_train)
    
    # Train Diffusion Model (Simplified for loop, using fewer epochs)
    diffusion_model = ConditionalDiffusionModel(target_dim, context_dim).to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=1e-3)
    diffusion_proc = DiffusionProcess(device=device)
    criterion = torch.nn.MSELoss()
    
    logger.info(f"Training Diffusion Model for L={L}, H={H}...")
    diffusion_model.train()
    for epoch in range(15): # Short epochs for loop
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            ctx = X_b.view(X_b.shape[0], -1)
            t = torch.randint(0, diffusion_proc.num_steps, (X_b.shape[0],), device=device).long()
            y_noisy, noise = diffusion_proc.add_noise(y_b, t)
            
            t_input = t.float().unsqueeze(1)
            pred_noise = diffusion_model(y_noisy, t_input, ctx)
            loss = criterion(pred_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate on Validation set
    diffusion_model.eval()
    
    all_y_true = []
    all_rw = []
    all_hm = []
    all_ridge = []
    all_diff = []
    
    with torch.no_grad():
        for X_v, y_v in val_loader:
            all_y_true.append(y_v.numpy())
            
            # Baselines
            all_rw.append(rw.predict(X_v).numpy())
            all_hm.append(hm.predict(X_v).numpy())
            all_ridge.append(ridge.predict(X_v).numpy())
            
            # Diffusion Sampling (Average of samples)
            X_v_dev = X_v.to(device)
            ctx_v = X_v_dev.view(X_v_dev.shape[0], -1)
            
            # Sample (batch_size, target_dim)
            diff_sample = diffusion_proc.sample(diffusion_model, ctx_v, y_v.shape)
            all_diff.append(diff_sample.cpu().numpy())
            
    y_true = np.concatenate(all_y_true, axis=0)
    y_rw = np.concatenate(all_rw, axis=0)
    y_hm = np.concatenate(all_hm, axis=0)
    y_ridge = np.concatenate(all_ridge, axis=0)
    y_diff = np.concatenate(all_diff, axis=0)
    
    # Compile Results
    models = {
        'Random Walk': y_rw,
        'Historical Mean': y_hm,
        'Ridge': y_ridge,
        'Diffusion': y_diff
    }
    
    results = []
    
    # We use Random Walk as the standard baseline for DM test
    baseline_preds = models['Random Walk']
    
    for name, preds in models.items():
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        
        # DM Test comparing this model against the baseline (Random Walk)
        dm_stat, p_val = diebold_mariano_test(y_true, baseline_preds, preds, h=H)
        
        # If the model IS the baseline, DM stat is 0 and p-val 1
        if name == 'Random Walk':
            dm_stat, p_val = 0.0, 1.0
            
        results.append({
            'L': L,
            'H': H,
            'Model': name,
            'RMSE': rmse,
            'DM_Stat': dm_stat,
            'P_Value': p_val
        })
        
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Grid Search
    windows = [10, 21, 42]
    horizons = [1, 5]
    
    all_experiments = []
    for l in windows:
        for h in horizons:
            res = evaluate_models(l, h, device)
            all_experiments.extend(res)
            
    df_results = pd.DataFrame(all_experiments)
    
    out_dir = 'reports/tables'
    os.makedirs(out_dir, exist_ok=True)
    df_results.to_csv(f'{out_dir}/experiment_results.csv', index=False)
    logger.info(f"Successfully saved experiment results to {out_dir}/experiment_results.csv")
    
    # Quick display
    print(df_results.sort_values(by=['L', 'H', 'RMSE']))
