from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from captum.attr import IntegratedGradients

from src.models.diffusion.model import ConditionalDiffusionModel
from src.models.evaluation.data_loader import get_dataloaders

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


class XAIWrapper(torch.nn.Module):
    def __init__(self, model: ConditionalDiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, context: torch.Tensor, noisy_y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(noisy_y, t, context)


def compute_integrated_gradients(
    model: ConditionalDiffusionModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    wrapper = XAIWrapper(model).to(device)
    ig = IntegratedGradients(wrapper)
    X, y = next(iter(dataloader))
    X, y = X.to(device), y.to(device)
    context = X.view(X.shape[0], -1)
    t = torch.full((X.shape[0], 1), 10.0, device=device)
    noisy_y = torch.randn_like(y).to(device)

    attrs = []
    for i in range(y.shape[1]):
        attr = ig.attribute(context, additional_forward_args=(noisy_y, t), target=i)
        attrs.append(attr)
    return torch.stack(attrs).mean(dim=0)


def save_ig_plots(attr: torch.Tensor, L: int, feature_names: list[str], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    importance = attr.abs().mean(dim=0).view(L, -1).detach().cpu().numpy()

    plt.figure(figsize=(12, 8))
    sns.heatmap(importance, cmap=sns.color_palette("mako", as_cmap=True), xticklabels=feature_names)
    plt.title("Integrated Gradients Attribution Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Lag")
    plt.tight_layout()
    plt.savefig(outdir / "importance_heatmap.png", dpi=300)
    plt.close()

    agg = importance.sum(axis=0)
    order = np.argsort(agg)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(np.array(feature_names)[order][:10], agg[order][:10], color=sns.color_palette("colorblind", 1))
    plt.title("Top Attribution Features (Integrated Gradients)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Attribution magnitude")
    plt.tight_layout()
    plt.savefig(outdir / "top_features_bar.png", dpi=300)
    plt.close()


def compute_counterfactual_importance(
    model: ConditionalDiffusionModel,
    process: DiffusionProcess,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    n_samples: int = 10,
) -> dict[str, float]:
    """
    Compute minimal-change counterfactuals for feature importance.
    For each feature channel, find the smallest L1 perturbation to flip the prediction sign.
    """
    model.eval()
    # Placeholder implementation
    # For each sample, for each channel, optimize delta to flip sign with min ||delta||_1
    # Use PyTorch optim on the channel.
    # Return average importance as 1 / min_norm for each channel.
    importance = {"returns": 0.5, "mask": 0.3, "amihud": 0.2, "regime": 0.1}  # Placeholder
    return importance


def save_seed_importance_plots(seed_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cols = ["importance_returns", "importance_mask", "importance_amihud", "importance_regime"]
    if not all(c in seed_df.columns for c in cols):
        return

    grouped = seed_df.groupby(["L", "H"])[cols].mean().reset_index()
    if grouped.empty:
        return

    heat = grouped.copy()
    heat["Config"] = heat.apply(lambda r: f"L={int(r['L'])}, H={int(r['H'])}", axis=1)
    heat = heat.set_index("Config")[cols]

    plt.figure(figsize=(8, 4))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap=sns.color_palette("viridis", as_cmap=True))
    plt.title("Permutation Importance (Diffusion, averaged across seeds)")
    plt.tight_layout()
    plt.savefig(outdir / "importance_heatmap.png", dpi=300)
    plt.close()

    mean_imp = seed_df[cols].mean().rename(
        {
            "importance_returns": "returns",
            "importance_mask": "mask",
            "importance_amihud": "amihud",
            "importance_regime": "regime",
        }
    )
    plt.figure(figsize=(6, 4))
    plt.bar(mean_imp.index, mean_imp.values, color=sns.color_palette("colorblind", 4))
    plt.title("Mean Feature Importance Across Seeds")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "mean_importance_bar.png", dpi=300)
    plt.close()


def save_stability_plot(stability_df: pd.DataFrame, outdir: Path) -> None:
    if stability_df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=stability_df, x="H", y="spearman_rho", color=sns.color_palette("colorblind")[0])
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.title("XAI Stability Across Seeds (Spearman correlation)")
    plt.ylabel("Spearman rho")
    plt.tight_layout()
    plt.savefig(outdir / "stability_by_horizon.png", dpi=300)
    plt.close()


def main() -> None:
    outdir = Path("reports/figures/xai")
    seed_path = Path("reports/tables/xai_seed_sensitivity.csv")
    stability_path = Path("reports/tables/xai_stability.csv")

    if seed_path.exists():
        seed_df = pd.read_csv(seed_path)
        save_seed_importance_plots(seed_df, outdir)
        if stability_path.exists():
            save_stability_plot(pd.read_csv(stability_path), outdir)
        print("Generated XAI robustness plots from reports/tables/xai_*.csv")
        return

    # Fallback: classic IG from checkpoint if robust CSV not available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, H = 21, 5
    _, val_loader = get_dataloaders(L, H, shuffle_train=False)
    model = ConditionalDiffusionModel(target_dim=4, context_dim=L * 12)
    checkpoint = Path("reports/logs/best_diffusion_model.pt")
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    attr = compute_integrated_gradients(model, val_loader, device=device)
    feature_names = [f"{m}_{t}" for m in ["Ret", "Msk", "Ami"] for t in ["EWZ", "EWW", "ECH", "GXG"]]
    save_ig_plots(attr, L=L, feature_names=feature_names, outdir=outdir)
    print("Generated fallback integrated-gradients plots.")


if __name__ == "__main__":
    main()
