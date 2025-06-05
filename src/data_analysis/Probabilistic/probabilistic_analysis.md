# 4D Probabilistic Analysis & Diagnostics <a name="4d-probabilistic-analysis--diagnostics"></a>

## 1. File: `src/Data_Analysis/probabilistic_analysis.py`

> _Advanced probability‐based summary for every feature (numeric & categorical)._
> The goal is that **no manual column naming** is required—everything is inferred from the DataFrame’s dtypes. You can call it by passing in your cleaned/interim dataset and (optionally) your target column. The script will:

1. **Fit common univariate distributions** to each numeric column, ranking them by AIC.
2. Compute **Shannon entropy** for every column (categorical or numeric).
3. Compute **mutual information** (or F‑score) between each feature and a provided target.
4. Build **conditional probability tables** for each categorical vs. the target.
5. Perform a **Probability Integral Transform (PIT)** on every numeric column.
6. Offer a quick **quantile transform** (normal or uniform) of all numerics.
7. Provide a simple **Bayesian group comparison** (mean ± 95 % CI) for any categorical grouping vs. a numeric target.
8. Compute **predictive intervals** (empirical quantiles) for any numeric feature.
9. Fit a **Gaussian copula** on all numeric columns.
10. Draw a **QQ/PP plot** for any numeric feature.
11. Calculate **Mahalanobis distance** to flag multivariate outliers.
12. Return **permutation‐based feature importance** (forest classifier/regressor).
13. Plot a **histogram + fitted PDF** for any numeric feature, using its best‐AIC distribution.

Everything writes back to `reports/probabilistic/…` (CSV or PNG) so you can inspect results or embed them in a downstream notebook.

---

> **Tip:** Because everything is auto‐detected, you do not need to specify column names. The script inspects dtypes and runs all applicable steps. Once you’ve run Phase 3 (data preparation) and already have `data/interim/clean.parquet`, this one‐liner completes all advanced probabilistic diagnostics in bulk.

---
