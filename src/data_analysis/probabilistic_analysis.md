Below is a minimal “drop‐in” module you can add under `src/Data_Analysis/` (e.g. as `probabilistic_analysis.py`) and a corresponding README snippet showing how to hook it into Phase 4 (Advanced EDA). The goal is that **no manual column naming** is required—everything is inferred from the DataFrame’s dtypes. You can call it by passing in your cleaned/interim dataset and (optionally) your target column. The script will:

1. **Impute missing** values in _all numeric_ columns (MICE, KNN or simple).
2. **Fit common univariate distributions** to each numeric column, ranking them by AIC.
3. Compute **Shannon entropy** for every column (categorical or numeric).
4. Compute **mutual information** (or F‑score) between each feature and a provided target.
5. Build **conditional probability tables** for each categorical vs. the target.
6. Perform a **Probability Integral Transform (PIT)** on every numeric column.
7. Offer a quick **quantile transform** (normal or uniform) of all numerics.
8. Provide a simple **Bayesian group comparison** (mean ± 95 % CI) for any categorical grouping vs. a numeric target.
9. Compute **predictive intervals** (empirical quantiles) for any numeric feature.
10. Fit a **Gaussian copula** on all numeric columns.
11. Draw a **QQ/PP plot** for any numeric feature.
12. Calculate **Mahalanobis distance** to flag multivariate outliers.
13. Return **permutation‐based feature importance** (forest classifier/regressor).
14. Plot a **histogram + fitted PDF** for any numeric feature, using its best‐AIC distribution.

Everything writes back to `reports/probabilistic/…` (CSV or PNG) so you can inspect results or embed them in a downstream notebook.

---

## 1. File: `src/Data_Analysis/probabilistic_analysis.py`

### 4D Probabilistic Analysis & Diagnostics <a name="4d-probabilistic-analysis--diagnostics"></a>

> _Advanced probability‐based summary for every feature (numeric & categorical)._

We supply a single Python script—**`probabilistic_analysis.py`**—that, given your cleaned/interim DataFrame, automatically:

1. **Imputes missing numeric values** (MICE, KNN or simple mean).
2. **Fits common parametric distributions** (normal, lognormal, gamma, beta, Weibull) to each numeric column and selects the best by AIC + KS test.
3. **Computes Shannon entropy** on every column (categorical or numeric).
4. **Calculates mutual information** (classification MI or F‐score) between each feature and a provided `_target_`.
5. **Builds conditional probability tables** for every categorical feature vs. the target.
6. Performs a **Probability Integral Transform (PIT)** on each numeric column.
7. Runs a **Quantile Transformer** (normal or uniform) on all numeric columns.
8. Executes a **Bayesian group comparison** (mean ± 95 % CI) for each categorical vs. numeric target.
9. Computes **empirical predictive intervals** (lower/upper quantiles) for each numeric variable.
10. Fits a **Gaussian copula** on the numeric subspace and dumps its parameters.
11. Calculates **Mahalanobis distance** to flag potential multivariate outliers.
12. Trains a **RandomForest model** (classifier or regressor) and outputs permutation‐based feature importances.
13. Draws **QQ & PP plots** and **histogram + fitted PDF** for each numeric feature.

All outputs (CSVs, JSONs, PNGs) are saved under `reports/probabilistic/`.  
Simply run:

```bash
python -m Data_Analysis.probabilistic_analysis \
      --data data/interim/clean.parquet \
      --target is_churn \
      --impute-method mice \
      --quantile-output normal
```

#### Output directory structure:

```
reports/
└── probabilistic/
    ├── imputed_numeric.csv
    ├── best_fit_distributions.json
    ├── shannon_entropy.csv
    ├── mutual_info_scores.csv           (if --target provided)
    ├── cpt_<cat_col>.csv                (one file per categorical column)
    ├── pit_transformed.csv
    ├── quantile_transformed_normal.csv  (or _uniform.csv)
    ├── group_stats_<feature>.csv        (one per categorical feature vs. numeric target)
    ├── copula_params.json
    ├── mahalanobis_distances.csv
    ├── permutation_importance.csv        (if --target provided)
    ├── qqpp_<num_col>.png                (QQ/PP plots)
    └── diagnostic_<num_col>.png          (Histogram + fitted PDF)
```

> **Tip:** Because everything is auto‐detected, you do not need to specify column names. The script inspects dtypes and runs all applicable steps. Once you’ve run Phase 3 (data preparation) and already have `data/interim/clean.parquet`, this one‐liner completes all advanced probabilistic diagnostics in bulk.

---

````

---

### 3. How to Integrate into Your DVC Pipeline

If you have a `dvc.yaml` stage for “EDA_advance,” replace it with something like:

```yaml
stages:
  …
  eda_probabilistic:
    cmd: python -m Data_Analysis.probabilistic_analysis \
             --data data/interim/clean.parquet \
             --target ${params.target} \
             --impute-method ${params.eda.impute_method} \
             --quantile-output ${params.eda.quantile_output}
    deps:
      - data/interim/clean.parquet
      - src/Data_Analysis/probabilistic_analysis.py
    outs:
      - reports/probabilistic/imputed_numeric.csv
      - reports/probabilistic/best_fit_distributions.json
      - reports/probabilistic/shannon_entropy.csv
      - reports/probabilistic/mutual_info_scores.csv
      - reports/probabilistic/pit_transformed.csv
      - reports/probabilistic/quantile_transformed_${params.eda.quantile_output}.csv
      - reports/probabilistic/copula_params.json
      - reports/probabilistic/mahalanobis_distances.csv
      - reports/probabilistic/permutation_importance.csv
      - reports/probabilistic/qqpp_*.png
      - reports/probabilistic/diagnostic_*.png
````

And in your `params.yaml`:

```yaml
eda:
  impute_method: "mice"
  quantile_output: "normal"
```

That way, `dvc repro eda_probabilistic` will re‐run **only** this stage if anything under `src/Data_Analysis/probabilistic_analysis.py` or `data/interim/clean.parquet` changes.

---

## 4. Quick Verification (Toy Example)

To verify everything works “out of the box,” drop a tiny CSV into:

```
data/raw/toy.csv        # e.g. 10 rows with 2 numeric columns, 1 categorical, 1 target
```

Then:

```bash
# 1. Ingest toy data (Phase 2) – outputs `data/interim/clean.parquet`
python -m Data_Ingestion.omni_cli file data/raw/toy.csv --save

# 2. Prepare data (Phase 3 – minimal defaults)
python -m ml_pipeline.prepare

# 3. Run probabilistic diagnostics (Phase 4D)
python -m Data_Analysis.probabilistic_analysis --data data/interim/clean.parquet --target target_column
```

You should see `reports/probabilistic/…` populated with a handful of CSVs and PNGs within a few seconds.
