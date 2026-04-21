import torch
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.diffusion.model import ConditionalDiffusionModel

class XAIWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, context, noisy_y, t):
        # We attribute to context
        return self.model(noisy_y, t, context)

def compute_attributions(model, dataloader, device='cpu'):
    model.eval()
    wrapper = XAIWrapper(model).to(device)
    ig = IntegratedGradients(wrapper)
    
    all_attr = []
    
    # We'll take a subset for XAI to be faster
    X, y = next(iter(dataloader))
    X, y = X.to(device), y.to(device)
    
    # Flatten context
    context = X.view(X.shape[0], -1)
    
    # Diffusion parameters for evaluation
    t = torch.full((X.shape[0], 1), 50.0, device=device) # Mid-way diffusion step
    noisy_y = torch.randn_like(y).to(device)
    
    # Compute attribution for each asset target
    asset_attributions = []
    for i in range(y.shape[1]):
        attr = ig.attribute(context, additional_forward_args=(noisy_y, t), target=i)
        asset_attributions.append(attr)
        
    # Average attribution across assets
    mean_attr = torch.stack(asset_attributions).mean(dim=0)
    return mean_attr

def plot_importance(attr, L, feature_names, output_path='reports/figures/xai/importance.png'):
    # attr: [batch, L*N*F]
    # Reshape to [L, N*F]
    importance = attr.abs().mean(dim=0).view(L, -1).detach().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(importance, cmap='magma', xticklabels=feature_names, yticklabels=[f't-{L-i}' for i in range(L)])
    plt.title("Feature Importance Attribution (Integrated Gradients)")
    plt.xlabel("Features")
    plt.ylabel("Lag")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def calculate_emh_zscore(attr, threshold=2.5):
    importance_scores = attr.abs().mean(dim=0).detach().cpu().numpy()
    mean = np.mean(importance_scores)
    std = np.std(importance_scores) + 1e-8
    
    z_scores = (importance_scores - mean) / std
    max_z = np.max(z_scores)
    
    rejected = max_z > threshold
    return max_z, rejected

import os
if __name__ == "__main__":
    from src.models.evaluation.data_loader import get_dataloaders
    
    L, H = 21, 5
    _, val_loader = get_dataloaders(L, H)
    
    model = ConditionalDiffusionModel(target_dim=4, context_dim=L*12)
    checkpoint = 'reports/logs/best_diffusion_model.pt'
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
        
    attr = compute_attributions(model, val_loader)
    max_z, rejected = calculate_emh_zscore(attr)
    
    print(f"Max Z-Score: {max_z:.2f} | EMH Rejected: {rejected}")
    
    feat_names = [f"{m}_{t}" for m in ['Ret', 'Msk', 'Ami'] for t in ['EWZ', 'EWW', 'ECH', 'GXG']]
    plot_importance(attr, L, feat_names)
