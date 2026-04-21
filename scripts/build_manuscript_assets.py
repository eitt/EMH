#!/usr/bin/env python3
"""Build Overleaf-friendly manuscript assets (tables + figures)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path.cwd().resolve()
REPORTS_TABLES = ROOT / "reports" / "tables"
REPORTS_FIGURES = ROOT / "reports" / "figures"
OUT_TABLES = ROOT / "tables" / "generated"
OUT_FIGURES = ROOT / "figures" / "generated"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def fmt_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def write_table(
    path: Path,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    note: str | None = None,
) -> None:
    colspec = "l" + "r" * (len(columns) - 1)
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if note:
        lines.append(f"\\caption*{{\\footnotesize {note}}}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_figures() -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    figure_map = {
        REPORTS_FIGURES / "rmse_comparison.png": OUT_FIGURES / "fig_rmse_by_model.png",
        REPORTS_FIGURES / "pvalue_heatmap.png": OUT_FIGURES / "fig_dm_pvalue_heatmap.png",
        REPORTS_FIGURES / "diffusion_seed_sensitivity.png": OUT_FIGURES / "fig_diffusion_seed_sensitivity.png",
        REPORTS_FIGURES / "xai" / "importance_heatmap.png": OUT_FIGURES / "fig_xai_importance_heatmap.png",
        REPORTS_FIGURES / "xai" / "top_features_bar.png": OUT_FIGURES / "fig_xai_top_features.png",
    }
    for src, dst in figure_map.items():
        if src.exists():
            shutil.copy2(src, dst)


def build_data_overview() -> None:
    returns = pd.read_csv(ROOT / "data" / "processed" / "returns.csv", index_col=0, parse_dates=True)
    mask = pd.read_csv(ROOT / "data" / "processed" / "mask.csv", index_col=0, parse_dates=True)
    rows = [
        ["Assets", str(returns.shape[1])],
        ["Observations (daily)", str(returns.shape[0])],
        ["Start date", str(returns.index.min().date())],
        ["End date", str(returns.index.max().date())],
        ["Average active mask", fmt_float(mask.mean().mean(), 3)],
    ]
    write_table(
        OUT_TABLES / "tab_data_overview.tex",
        "Data overview.",
        "tab:data-overview",
        ["Item", "Value"],
        rows,
        note="Raw data source: Yahoo Finance via yfinance; assets: EWZ, EWW, ECH, GXG.",
    )


def build_summary_statistics() -> None:
    df = pd.read_csv(REPORTS_TABLES / "summary_statistics.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                latex_escape(str(r["ticker"])),
                fmt_float(r["mean"], 5),
                fmt_float(r["std"], 5),
                fmt_float(r["skew"], 3),
                fmt_float(r["kurtosis"], 3),
                fmt_float(r["acf1"], 3),
            ]
        )
    write_table(
        OUT_TABLES / "tab_summary_statistics.tex",
        "Summary statistics of daily log returns.",
        "tab:summary-statistics",
        ["ETF", "Mean", "Std. Dev.", "Skewness", "Kurtosis", "ACF(1)"],
        rows,
    )


def build_diagnostics() -> None:
    df = pd.read_csv(REPORTS_TABLES / "diagnostic_tests.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                latex_escape(str(r["ticker"])),
                fmt_float(r["adf_price_p"], 3),
                fmt_float(r["adf_return_p"], 3),
                fmt_float(r["lb10_return_p"], 3),
                fmt_float(r["lb10_sqreturn_p"], 3),
            ]
        )
    write_table(
        OUT_TABLES / "tab_diagnostics.tex",
        "Stationarity and serial dependence diagnostics.",
        "tab:diagnostics",
        ["ETF", "ADF p (price)", "ADF p (return)", "LB p (return)", "LB p (return$^2$)"],
        rows,
        note="ADF is Augmented Dickey-Fuller test; LB is Ljung-Box at lag 10.",
    )


def build_return_correlations() -> None:
    df = pd.read_csv(REPORTS_TABLES / "return_correlations.csv")
    tickers = [str(c) for c in df.columns if c != "ticker"]
    rows = []
    for _, r in df.iterrows():
        row = [latex_escape(str(r["ticker"]))]
        for t in tickers:
            row.append(fmt_float(r[t], 3))
        rows.append(row)
    write_table(
        OUT_TABLES / "tab_return_correlations.tex",
        "Pairwise correlation matrix of daily log returns.",
        "tab:return-correlations",
        ["ETF"] + tickers,
        rows,
    )


def build_rmse() -> None:
    df = pd.read_csv(REPORTS_TABLES / "rmse_pivot.csv")
    model_cols = [c for c in df.columns if c not in {"L", "H"}]
    rows = []
    for _, r in df.iterrows():
        row = [str(int(r["L"])), str(int(r["H"]))]
        row.extend(fmt_float(r[c], 5) for c in model_cols)
        rows.append(row)
    write_table(
        OUT_TABLES / "tab_rmse_by_config.tex",
        "Out-of-sample RMSE by lookback window and forecast horizon.",
        "tab:rmse-by-config",
        ["L", "H"] + model_cols,
        rows,
        note="Lower values indicate better forecast accuracy.",
    )


def build_performance_summary() -> None:
    df = pd.read_csv(REPORTS_TABLES / "experiment_results.csv")
    rows = []
    for _, r in df.sort_values(["L", "H", "RMSE"]).iterrows():
        rows.append(
            [
                str(int(r["L"])),
                str(int(r["H"])),
                latex_escape(str(r["Model"])),
                fmt_float(r["RMSE"], 5),
                fmt_float(r["MAE"], 5),
                fmt_float(r["Directional_Accuracy"], 3),
                str(int(r["N_Windows"])),
            ]
        )
    write_table(
        OUT_TABLES / "tab_model_performance_summary.tex",
        "Average losses and directional accuracy across expanding windows.",
        "tab:model-performance-summary",
        ["L", "H", "Model", "RMSE", "MAE", "Dir. Acc.", "Windows"],
        rows,
    )


def build_dm_summary() -> None:
    df = pd.read_csv(REPORTS_TABLES / "dm_tests.csv")
    agg = (
        df.groupby("Model", as_index=False)
        .agg(mean_dm=("DM_Stat", "mean"), sig_count=("significant_5pct", "sum"), configs=("significant_5pct", "size"))
        .sort_values("Model")
    )
    rows = []
    for _, r in agg.iterrows():
        rows.append(
            [
                latex_escape(str(r["Model"])),
                fmt_float(r["mean_dm"], 3),
                f"{int(r['sig_count'])} / {int(r['configs'])}",
            ]
        )
    write_table(
        OUT_TABLES / "tab_dm_summary.tex",
        "Diebold-Mariano comparison versus random walk.",
        "tab:dm-summary",
        ["Model", "Mean DM statistic", "Significant at 5\\%"],
        rows,
        note="Negative DM means the candidate model has higher loss than random walk.",
    )


def build_robustness_summary() -> None:
    path = REPORTS_TABLES / "robustness_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = [[latex_escape(str(r["item"])), latex_escape(str(r["value"]))] for _, r in df.iterrows()]
    write_table(
        OUT_TABLES / "tab_robustness_summary.tex",
        "Evaluation and leakage-control summary.",
        "tab:robustness-summary",
        ["Item", "Value"],
        rows,
    )


def build_diffusion_ablation() -> None:
    path = REPORTS_TABLES / "diffusion_ablation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.sort_values(["L", "H", "rmse_mean"]).iterrows():
        rows.append(
            [
                str(int(r["L"])),
                str(int(r["H"])),
                latex_escape(str(r["feature_set"])),
                fmt_float(r["rmse_mean"], 5),
                fmt_float(r["mae_mean"], 5),
                fmt_float(r["rmse_std"], 5),
            ]
        )
    write_table(
        OUT_TABLES / "tab_diffusion_ablation.tex",
        "Diffusion sensitivity to feature-set ablations across seeds.",
        "tab:diffusion-ablation",
        ["L", "H", "Feature set", "RMSE", "MAE", "RMSE std."],
        rows,
    )


def build_seed_sensitivity() -> None:
    path = REPORTS_TABLES / "diffusion_seed_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.sort_values(["L", "H", "feature_set", "seed"]).iterrows():
        rows.append(
            [
                str(int(r["L"])),
                str(int(r["H"])),
                latex_escape(str(r["feature_set"])),
                str(int(r["seed"])),
                fmt_float(r["rmse_mean"], 5),
                fmt_float(r["mae_mean"], 5),
            ]
        )
    write_table(
        OUT_TABLES / "tab_diffusion_seed_sensitivity.tex",
        "Diffusion seed-level sensitivity by horizon and feature set.",
        "tab:diffusion-seed-sensitivity",
        ["L", "H", "Feature set", "Seed", "RMSE", "MAE"],
        rows,
    )


def build_xai_sensitivity() -> None:
    path = REPORTS_TABLES / "xai_seed_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = [
        [
            str(int(r["L"])),
            str(int(r["H"])),
            str(int(r["seed"])),
            fmt_float(r["max_z_score"], 3),
            "Yes" if bool(r["reject_threshold_2p5"]) else "No",
        ]
        for _, r in df.sort_values(["L", "H", "seed"]).iterrows()
    ]
    write_table(
        OUT_TABLES / "tab_xai_seed_sensitivity.tex",
        "Permutation-importance max z-score sensitivity across diffusion seeds.",
        "tab:xai-seed-sensitivity",
        ["L", "H", "Seed", "Max z-score", "Above 2.5 threshold"],
        rows,
    )


def build_xai_stability() -> None:
    path = REPORTS_TABLES / "xai_stability.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.sort_values(["L", "H", "seed_a", "seed_b"]).iterrows():
        rows.append(
            [
                str(int(r["L"])),
                str(int(r["H"])),
                f"{int(r['seed_a'])}-{int(r['seed_b'])}",
                fmt_float(r["spearman_rho"], 3),
            ]
        )
    write_table(
        OUT_TABLES / "tab_xai_stability.tex",
        "XAI stability across seed pairs (Spearman correlation).",
        "tab:xai-stability",
        ["L", "H", "Seed pair", "Spearman rho"],
        rows,
    )


def main() -> int:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    copy_figures()
    build_data_overview()
    build_summary_statistics()
    build_diagnostics()
    build_return_correlations()
    build_rmse()
    build_performance_summary()
    build_dm_summary()
    build_robustness_summary()
    build_diffusion_ablation()
    build_seed_sensitivity()
    build_xai_sensitivity()
    build_xai_stability()

    print(f"Generated tables in: {OUT_TABLES}")
    print(f"Generated figures in: {OUT_FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
