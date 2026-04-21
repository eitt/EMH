# Asset Pipeline (Robust + Overleaf-Oriented)

## 1) Folder organization

```text
.
|- data/
|  |- raw/
|  `- processed/
|- reports/
|  |- figures/              # analysis figures from pipeline
|  `- tables/               # CSV diagnostics/results/robustness outputs
|- scripts/
|  |- run_pipeline.py
|  |- build_manuscript_assets.py
|  `- build_working_paper.py
|- figures/
|  |- generated/            # manuscript-ready standardized figure names
|  `- static/               # optional manually curated figures
|- tables/
|  |- generated/            # manuscript-ready LaTeX tables
|  `- static/               # optional manually curated tables
|- tex/
|  |- main.tex
|  `- sections/
|- paper/
|  `- references.bib
`- output/
   `- working_paper/        # Overleaf-ready bundle + manifest
```

## 2) Experiment and manuscript pipeline

### End-to-end

```bash
python scripts/run_pipeline.py
```

This executes:
1. preprocessing,
2. robust experiment loop,
3. figures,
4. XAI robustness plots,
5. manuscript asset generation,
6. Overleaf bundle generation.

### Optional skip flags

```bash
python scripts/run_pipeline.py --skip-train
python scripts/run_pipeline.py --skip-xai
python scripts/run_pipeline.py --skip-manuscript-assets
python scripts/run_pipeline.py --skip-working-paper
```

## 3) Robustness outputs storage

Core robust outputs in `reports/tables/` include:
- `experiment_results.csv`
- `per_window_losses.csv`
- `dm_tests.csv`
- `rmse_pivot.csv`
- `leakage_checks.csv`
- `robustness_summary.csv`
- `diffusion_training_stability.csv`
- `diffusion_seed_sensitivity.csv`
- `diffusion_ablation.csv`
- `xai_seed_sensitivity.csv`
- `xai_stability.csv`
- plus descriptive diagnostics (`summary_statistics.csv`, `diagnostic_tests.csv`, `return_correlations.csv`)

## 4) Manuscript asset generation

`scripts/build_manuscript_assets.py` transforms report CSVs and figures into:

### `tables/generated/`
- `tab_data_overview.tex`
- `tab_summary_statistics.tex`
- `tab_diagnostics.tex`
- `tab_return_correlations.tex`
- `tab_rmse_by_config.tex`
- `tab_model_performance_summary.tex`
- `tab_dm_summary.tex`
- `tab_robustness_summary.tex`
- `tab_diffusion_ablation.tex`
- `tab_diffusion_seed_sensitivity.tex`
- `tab_xai_seed_sensitivity.tex`
- `tab_xai_stability.tex`

### `figures/generated/`
- `fig_rmse_by_model.png`
- `fig_dm_pvalue_heatmap.png`
- `fig_diffusion_seed_sensitivity.png`
- `fig_xai_importance_heatmap.png`
- `fig_xai_top_features.png`

## 5) Working-paper bundle for Overleaf

`scripts/build_working_paper.py` builds:
- `output/working_paper/bundle/` with `tex/`, `tables/generated/`, `figures/generated/`, `paper/references.bib`
- `output/working_paper/manifest.json` with article-style and theory-citation checks

## 6) Overleaf refresh steps

1. Run `python scripts/run_pipeline.py` locally.
2. Upload `output/working_paper/bundle/*` to Overleaf.
3. Compile `tex/main.tex` with BibTeX sequence:
   - `pdflatex`
   - `bibtex`
   - `pdflatex`
   - `pdflatex`

## 7) Remaining manual steps

- Verify final bibliography build in Overleaf (local MiKTeX may fail if BibTeX setup is restricted).
- Optional: perform final wording polish in `tex/sections/` after the latest regenerated tables/figures.
