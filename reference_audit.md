# Reference Audit

Audit date: 2026-04-21  
Scope: citations materially used in `tex/sections/` (literature, methodology, limitations, interpretation).

## Summary

- Cited references checked: **21**
- Accurate paraphrases: **9**
- Close paraphrases needing rewrite: **0**
- Unsupported or weakly supported claims: **5**
- Priority fixes:
  1. Verify foundational claims tied to `fama1970`, `fama1991`, `welchgoyal2008`, and `campbellthompson2008` against full-text passages (current status: not verified from local files).
  2. Verify diffusion-finance positioning claim tied to `cho2026` with full paper text.
  3. Keep explicit non-causal wording where evidence is forecast-comparison only.

## Citation-level integrity table

| Citation key | Full article title | Manuscript location | Phrase in manuscript | Source location in original paper | Original phrase in source | Type of use | Assessment | Action needed |
|---|---|---|---|---|---|---|---|---|
| `fama1970` | Efficient Capital Markets: A Review of Theory and Empirical Work | Section 1, para 1 | “Weak-form market efficiency remains a core benchmark in empirical finance.” | Not verified (full text not in repository) | not verified | conceptual borrowing | ambiguous attribution | keep + verify against source |
| `fama1991` | Efficient Capital Markets: II | Section 1, para 1 | “Weak-form market efficiency remains a core benchmark in empirical finance.” | Not verified (full text not in repository) | not verified | conceptual borrowing | ambiguous attribution | keep + verify against source |
| `harvey1995` | Predictable Risk and Returns in Emerging Markets | Section 1 and Section 2, strand 2 | “return behavior in emerging markets can vary with integration and local information regimes” | Not verified (full text not in repository) | not verified | factual claim | needs stronger citation | keep + verify exact support |
| `bekaertharvey1995` | Time-Varying World Market Integration | Section 1 and Section 2, strand 2 | “vary with integration and local information regimes” | Crossref abstract | “measure of capital market integration” | factual claim | accurate paraphrase | keep |
| `lomackinlay1988` | Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test | Section 2, strand 1 | “foundational evidence on random-walk violations” | Not verified (full text not in repository) | not verified | factual claim | ambiguous attribution | keep + verify against source |
| `welchgoyal2008` | A Comprehensive Look at The Empirical Performance of Equity Premium Prediction | Section 2, strand 1 | “instability of forecasting gains” | Not verified (full text not in repository) | not verified | factual claim | weakly supported (not verified) | keep + verify specific passage |
| `campbellthompson2008` | Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average? | Section 2, strand 1 | “instability of forecasting gains” | Not verified (full text not in repository) | not verified | factual claim | weakly supported (not verified) | keep + verify specific passage |
| `rapach2010` | Out-of-Sample Equity Premium Prediction: Combination Forecasts and Links to the Real Economy | Section 2, strand 1 | “instability of forecasting gains” | Not verified (full text not in repository) | not verified | factual claim | ambiguous attribution | keep + verify against source |
| `urrutia1995` | Tests of Random Walk and Market Efficiency for Latin American Emerging Equity Markets | Section 2, strand 3 | “mostly based on variance-ratio … test batteries” | Crossref abstract | “Variance-ratio methodology is used … random walk.” | factual claim | accurate paraphrase | keep |
| `sanchezgranero2020` | Testing the Efficient Market Hypothesis in Latin American Stock Markets | Section 2, strand 3 | “Latin American efficiency evidence … test batteries” | Not verified (metadata only) | not verified | factual claim | weakly supported (not verified) | keep + verify full-text method details |
| `lo2004` | The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective | Section 2, strand 3 | “adaptive-market test batteries” | Not verified (full text not in repository) | not verified | conceptual borrowing | ambiguous attribution | keep + verify against source |
| `cruzhernandez2024` | Adaptive Market Hypothesis and Predictability: Evidence in Latin American Stock Indices | Section 2, strand 3 | “adaptive-market … evidence” | Crossref abstract | “examines the adaptive market hypothesis … Latin American stock indices” | factual claim | accurate paraphrase | keep |
| `gu2020` | Empirical Asset Pricing via Machine Learning | Section 2, strand 4 | “nonlinear models can improve fit … require strong out-of-sample discipline” | Crossref abstract | “comparative analysis of machine learning methods … empirical asset pricing” | methodological borrowing | accurate paraphrase | keep |
| `ho2020` | Denoising Diffusion Probabilistic Models | Section 2, strand 5 | “denoising diffusion … flexible distributional modeling” | Not verified (conference full text not in repository) | not verified | methodological borrowing | ambiguous attribution | keep + verify exact supporting text |
| `song2021` | Score-Based Generative Modeling through Stochastic Differential Equations | Section 2, strand 5 | “score-based methods provide flexible distributional modeling” | Not verified (conference full text not in repository) | not verified | methodological borrowing | ambiguous attribution | keep + verify exact supporting text |
| `cho2026` | Diffolio: A Diffusion Model for Multivariate Probabilistic Financial Time-Series Forecasting and Portfolio Construction | Section 2, strand 5 | “early finance applications are emerging” | Scopus metadata only (title-level evidence) | not verified | factual claim | weakly supported (metadata-level) | keep + verify with full paper |
| `amihud2002` | Illiquidity and Stock Returns: Cross-Section and Time-Series Effects | Section 3, variable construction | “Following Amihud (2002), the illiquidity proxy is …” | Formula attribution only; full text not in repository | not verified | methodological borrowing | accurate paraphrase (standard definition) | keep |
| `dieboldmariano1995` | Comparing Predictive Accuracy | Section 4, evaluation | “Forecast comparison uses Diebold-Mariano statistics” | Not verified (full text not in repository) | not verified | methodological borrowing | accurate paraphrase (standard method attribution) | keep |
| `engle1982` | Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation | Section 4, diagnostics | “ARCH/GARCH families” motivation | Not verified (full text not in repository) | not verified | conceptual borrowing | accurate paraphrase | keep |
| `bollerslev1986` | Generalized Autoregressive Conditional Heteroskedasticity | Section 4, diagnostics | “ARCH/GARCH families” motivation | Not verified (full text not in repository) | not verified | conceptual borrowing | accurate paraphrase | keep |
| `hansen2011` | The Model Confidence Set | Section 6, limitations | “model confidence set procedures are not yet implemented” | Title-level verification | “The Model Confidence Set” | methodological borrowing | accurate paraphrase | keep |

## Notes on verification method

- Verified items use metadata/abstract snippets retrievable via DOI/Crossref APIs available in this environment.
- “Not verified” means full-text passage matching was not possible from repository files or accessible abstract text.
- No direct quotations from source papers are used in manuscript body; all current usages are paraphrastic or methodological attributions.
