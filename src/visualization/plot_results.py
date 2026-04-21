import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set publication quality aesthetics
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
# Muted, colorblind-friendly colors (blue, orange, green, red)
COLORS = sns.color_palette("colorblind", 4)

def load_results(path='reports/tables/experiment_results.csv'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)

def plot_rmse_bar(df):
    plt.figure(figsize=(10, 6))
    
    # We create a composite hue to group by L and Model
    # Simple bar plot across H
    g = sns.catplot(
        data=df, kind="bar",
        x="H", y="RMSE", hue="Model", col="L",
        palette=COLORS, height=5, aspect=0.8,
        errorbar=None # Since we only have one run per config in the loop right now
    )
    
    g.set_axis_labels("Forecast Horizon (H)", "RMSE (Log Returns)")
    g.set_titles("Lookback Window (L) = {col_name}")
    g.despine(left=True)
    
    out_path = 'reports/figures/rmse_comparison.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_pvalue_heatmap(df):
    """Plots a heatmap of DM Test p-values vs Random Walk"""
    plt.figure(figsize=(8, 6))
    
    # Filter out Random Walk since it's the baseline (p=1.0)
    df_compare = df[df['Model'] != 'Random Walk'].copy()
    
    # Create pivot for heatmap (L, H) vs Model P-Value
    # We will combine L and H for the Y axis
    df_compare['Config'] = df_compare.apply(lambda r: f"L={r['L']}, H={r['H']}", axis=1)
    
    pivot = df_compare.pivot(index='Config', columns='Model', values='P_Value')
    
    # We want to highlight significance < 0.05
    # Use a diverging colormap where low is red (significant), high is white/blue
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    
    ax = sns.heatmap(pivot, annot=True, cmap=cmap, vmin=0.0, vmax=1.0, 
                     cbar_kws={'label': 'Diebold-Mariano p-value\n(vs Random Walk)'})
    
    ax.set_title("Statistical Significance of Forecast Improvements\n(p < 0.05 indicates model beats Random Walk)")
    
    out_path = 'reports/figures/pvalue_heatmap.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        df = load_results()
        plot_rmse_bar(df)
        plot_pvalue_heatmap(df)
        print("Generated scientific visualizations in reports/figures/")
    except Exception as e:
        print(f"Error generating plots: {e}")
