from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_results(
    experiment_path: str = "reports/tables/experiment_results.csv",
    per_window_path: str = "reports/tables/per_window_losses.csv",
    dm_path: str = "reports/tables/dm_tests.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Missing {experiment_path}")
    exp = pd.read_csv(experiment_path)
    per_window = pd.read_csv(per_window_path) if os.path.exists(per_window_path) else pd.DataFrame()
    dm = pd.read_csv(dm_path) if os.path.exists(dm_path) else pd.DataFrame()
    return exp, per_window, dm


def plot_rmse_box(per_window: pd.DataFrame, experiment: pd.DataFrame) -> None:
    out_path = "reports/figures/rmse_comparison.png"
    _ensure_dir(out_path)

    if per_window.empty:
        pivot = experiment.pivot_table(index=["L", "H"], columns="Model", values="RMSE")
        fig, ax = plt.subplots(figsize=(11, 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("RMSE")
        ax.set_title("Average RMSE by configuration")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return

    df = per_window.copy()
    df["Config"] = df.apply(lambda r: f"L={int(r['L'])}, H={int(r['H'])}", axis=1)
    order = sorted(df["Config"].unique(), key=lambda s: (int(s.split(",")[0].split("=")[1]), int(s.split("=")[2])))
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x="Config", y="RMSE", hue="model", ax=ax)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("RMSE")
    ax.set_title("Per-window RMSE distribution by model")
    ax.tick_params(axis="x", rotation=30)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_pvalue_heatmap(dm: pd.DataFrame) -> None:
    out_path = "reports/figures/pvalue_heatmap.png"
    _ensure_dir(out_path)

    if dm.empty:
        return
    df = dm.copy()
    df = df[df["Model"] != "Random Walk"]
    df["Config"] = df.apply(lambda r: f"L={int(r['L'])}, H={int(r['H'])}", axis=1)
    pivot = df.pivot(index="Config", columns="Model", values="P_Value")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap=sns.diverging_palette(10, 240, as_cmap=True),
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "DM p-value vs Random Walk"},
        ax=ax,
    )
    ax.set_title("Diebold-Mariano p-values (lower is stronger evidence against equal loss)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_seed_sensitivity() -> None:
    in_path = "reports/tables/diffusion_seed_sensitivity.csv"
    out_path = "reports/figures/diffusion_seed_sensitivity.png"
    if not os.path.exists(in_path):
        return
    _ensure_dir(out_path)
    df = pd.read_csv(in_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=df,
        x="seed",
        y="rmse_mean",
        hue="feature_set",
        style="H",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Diffusion RMSE sensitivity across seeds, horizons, and feature sets")
    ax.set_ylabel("Mean RMSE")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    exp, per_window, dm = load_results()
    plot_rmse_box(per_window, exp)
    plot_pvalue_heatmap(dm)
    plot_seed_sensitivity()
    print("Generated scientific visualizations in reports/figures/")
