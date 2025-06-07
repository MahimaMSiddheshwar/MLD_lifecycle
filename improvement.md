# Machine Learning Pipeline Reference

> **Purpose:**  
> This document codifies a reproducible, configurable, end-to-end ML pipeline from raw data ingestion through feature engineering — pulling together best practices from Stanford, MIT, Harvard and our in-house improvements.
>
> **How to use:**  
> For each project, walk through the numbered stages below. Each stage lists **Objectives**, **Key Tasks**, **Checks & Metrics**, and **Configurable Parameters**.

---

## Table of Contents

1. [Stage 0: Project Setup & Configuration](#stage-0-project-setup--configuration)
2. [Stage 1: Data Ingestion](#stage-1-data-ingestion)
3. [Stage 2: Data Validation & Sanity Checks](#stage-2-data-validation--sanity-checks)
4. [Stage 3: Missingness & Cleaning](#stage-3-missingness--cleaning)
5. [Stage 4: Outlier Detection & Treatment](#stage-4-outlier-detection--treatment)
6. [Stage 5: Scaling & Transformation](#stage-5-scaling--transformation)
7. [Stage 6: Feature Engineering & Selection](#stage-6-feature-engineering--selection)
8. [Stage 7: Train/Val/Test Split & Baseline](#stage-7-trainvaltest-split--baseline)
9. [Stage 8: Leakage Detection & Integrity](#stage-8-leakage-detection--integrity)
10. [Next Steps](#next-steps)

---

## Stage 0: Project Setup & Configuration

### Objectives

- Centralize all thresholds, file paths, random seeds, reporting options.
- Ensure reproducibility (version your code, record git SHA).

### Key Tasks

- Create `config.yaml` (or Python dict) with:

  ```yaml
  data:
    raw_path: data/raw/
    proc_path: data/processed/
  thresholds:
    high_cardinality_ratio: 0.10
    missing_frac_drop: 0.90
    outlier_vote_threshold: 3
    nzv_unique_threshold: 2
    skew_robust: 1.0
    kurtosis_robust: 5.0
    shapiro_p_thresh: 0.05
  random_seed: 42
  ```

- Record experiment metadata: git commit, timestamp, config hash.

### Checks & Metrics

- Validate `config.yaml` schema (e.g. with [Cerberus](https://docs.python-cerberus.org/)).
- Halt if any required key is missing.

---

## Stage 1: Data Ingestion

### Objectives

- Load raw tables (CSV, Parquet, database) into a canonical in-memory format.
- Persist a snapshot (Parquet) for downstream stages.

### Key Tasks

1. **Read source** into `pd.DataFrame` (or Spark/DB connector).
2. **Schema enforcement**:

   - Define expected dtypes & nullability in a schema file (JSON/Avro/tdl).
   - Cast or reject mismatches.

3. **Write** cleaned snapshot to `data/processed/data.parquet`.

### Checks & Metrics

- Row counts vs. source logs.
- Column count & dtypes match schema.
- Checksum Parquet file for integrity.

---

## Stage 2: Data Validation & Sanity Checks

> _Inspired by Harvard’s “Data Quality Framework”._

### Objectives

- Detect structural, semantic, and cross-field anomalies before heavy processing.

### Key Tasks & Checks

1. **Mixed-Type Detection**

   - For each column:

     ```python
     types = df[col].dropna().map(type).value_counts()
     if len(types) > 1: report.mixed_types[col] = types.to_dict()
     ```

2. **Impossible Value Rules**

   - User-supplied forbidden sets (e.g. negative ages, zeros in IDs).

3. **Unexpected High-Cardinality**

   - `unique_ratio = nunique / n_rows`; if `> config.thresholds.high_cardinality_ratio`, flag.

4. **Near-Zero-Variance**

   - Drop if `nunique < config.thresholds.nzv_unique_threshold`.

5. **Duplicate Rows**

   - `df.duplicated().sum()`. Optionally remove.

6. **Custom Missing Markers**

   - Scan short strings (`len<=2`) with `freq > 1%` → treat as null.

---

## Stage 3: Missingness & Cleaning

> _Leverage MIT’s “Robust Missing Data” guidelines._

### Objectives

- Characterize missingness mechanism & apply robust imputation or flagging.

### Key Tasks

1. **Missingness Pattern Analysis**

   - Little’s omnibus MCAR test & per-column logistic test.

2. **Stratified Missingness**

   - `df.groupby(target)[col].apply(lambda s: s.isna().mean())`.

3. **Drop vs Impute**

   - Drop if `missing_frac > config.thresholds.missing_frac_drop`.
   - Else impute—choose strategy per column:

     - Numeric: mean/median/KNN (only if `n_rows <= knn_max_rows`) /random.
     - Categorical: mode/constant/random via TVD.

4. **Cast Mixed-Type**

   - If strings in numeric column ≥90% digit-like, cast to numeric.

### Checks & Metrics

- Report per-column missing fractions & chosen method.
- Report covariate shift pre/post imputation.

---

## Stage 4: Outlier Detection & Treatment

### Objectives

- Detect both univariate & multivariate outliers, then either cap (winsorize) or drop.

### Key Tasks

1. **Univariate Rules** (each gives 1 vote)

   - IQR (1.5×), Z-score (|z|>3), ModZ (|modz|>3.5), Tukey (2× IQR), 1st/99th pct.

2. **Multivariate Rules** (row-complete only)

   - If `n_rows >= 5×n_features` & ≥60% Gaussian: Mahalanobis (χ² 97.5%).
   - Else if `n_rows<2k & n_features<50`: LOF (novelty).
   - Else: IsolationForest (contamination 0.01).
   - Special case: if `n_features ≥ n_rows`, skip directly to IsolationForest.

3. **Voting & Threshold**

   - Flag real outliers if votes ≥ `config.thresholds.outlier_vote_threshold`.

4. **Treatment**

   - If `cap_outliers=True`: Winsorize at each column’s 1st/99th pct.
   - Else: drop rows.

### Checks & Metrics

- Number & fraction of rows flagged by each method.
- Final outlier count & treatment summary.

---

## Stage 5: Scaling & Transformation

> _Based on MIT’s “Adaptive Scaling” paper._

### Objectives

- Drop near-zero-variance, choose an optimal scaler, then apply extra transforms to nudge toward normality.

### Key Tasks

1. **Drop NZV Columns** (`nunique < config.nzv_threshold`).
2. **Choose Scaler** by skew/kurtosis:

   - If any `|skew| > sk_thresh_robust` or `|kurt| > kurt_thresh_robust` → RobustScaler
   - Else if ALL `|skew| < sk_thresh_standard` → StandardScaler
   - Else → MinMaxScaler

3. **Fit & Apply Scaler** → `self.scaler`.
4. **Post-Scale Normality Check** (Shapiro p > shapiro_p_thresh & |skew|\<transform_skew_thresh).
5. **Extra Transforms** (if not “Gaussian enough”):

   - Trial: none, Box-Cox (if all >0), Yeo-Johnson, Quantile→Normal
   - Choose by lexicographic `(pval, −|skew|)`.

6. **Persist** `scaler`, per-col transformer for `transform()`.

### Checks & Metrics

- Report chosen scaler & per-column transform with scores.
- JSON or DataFrame–style report for drift monitoring.

---

## Stage 6: Feature Engineering & Selection

> _Following Stanford’s “Feature Lab” guidelines._

### Objectives

- Add generic derived features, prune redundancy, preselect for modeling.

### Key Tasks

1. **Interaction Candidates**

   - Optionally generate pairwise (`A×B`, `A+B`) for top N numeric pairs by correlation.

2. **Binning / Grouping**

   - Auto-bin numeric into quartiles or quantile buckets.

3. **Categorical Encoding**

   - Target or frequency encoding for high-cardinality (>10 levels).

4. **Correlated Feature Pruning**

   - Drop one of any pair with `|corr| > config.thresholds.drop_corr` (e.g. 0.95).

5. **Recursive Feature Elimination (RFE)**

   - Light wrapper using your chosen estimator to preselect top K features.

6. **Save** transformation summary for `transform()`.

### Checks & Metrics

- Final feature count.
- Variance Inflation Factor (VIF) to detect multicollinearity.
- Permutation‐importance snapshot.

---

## Stage 7: Train/Val/Test Split & Baseline

> _CRISP-DM Step: Modeling, but here only splitting & baseline._

### Objectives

- Create reproducible splits; optionally oversample; compute trivial baselines.

### Key Tasks

1. **Parquet Splits**

   - Train/Val/Test: 60/20/20 stratified on target if requested.

2. **Oversampling**

   - If classification (`y.dtype.kind in {'O','i','b'}`) & `oversample=True`, apply SMOTE on train only.

3. **Baseline Models**

   - Regression: mean predictor → MAE, R²
   - Classification: majority class → accuracy, F1
   - (Optionally add k-NN vs naive Bayes as extra baselines.)

4. **Sanity Checks**

   - No index overlap between splits.
   - No feature identical to target.

### Checks & Metrics

- Row counts per split.
- Baseline metric summary in JSON.

---

## Stage 8: Leakage Detection & Integrity

### Objectives

- Catch both target-leakage and train/test separation leakage.

### Key Tasks

1. **Target-Leakage**

   - For each feature: compute AUC( feature → target ) > auc_thresh (e.g. 0.99)
   - Categorical: one-hot & take max AUC.

2. **Train/Test Separation**

   - Concat train/test, label 0/1, compute AUC( feature → is_train ).

3. **Row Overlap**

   - Merge on all features – if any exact duplicate, warn.

4. **Unseen Categories**

   - For each cat col: test_vals – train_vals → report.

### Checks & Metrics

- List of leaky features.
- Separation AUC per feature.
- Overlap row count.

---

## Next Steps & MLOps

1. **Model Selection & Tuning**

   - Wrap with [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/pipeline.html), grid/RandomSearchCV.

2. **Experiment Tracking**

   - Log metrics/params to MLflow or Weights & Biases.

3. **Model Validation**

   - Cross-validation, calibration curves, partial dependence.

4. **Deployment**

   - Package as API (FastAPI), containerize (Docker), auto-retrain triggers.

5. **Monitoring**

   - Periodic data-drift & prediction-drift checks, alerting on threshold breaches.

---

> **References & Further Reading**
>
> - CRISP-DM (SPC Whitepaper)
> - TDSP (Microsoft’s Team Data Science Process)
> - “Feature Lab” at Stanford CS229
> - MIT 6.86x “Machine Learning on Big Data” lecture notes
> - Harvard Data Science Common Workflow

_(This README is meant as a living document – feel free to iterate over it as your team’s needs evolve.)_

Here’s a **single, self-contained** `improvements.md` capturing every review point and extending them with additional suggestions. Feel free to drop it into your repo (e.g. at `docs/improvements.md`) and iterate from there.

````markdown
# Pipeline Code Review & Improvement Roadmap

> **Scope:** Six modules (Stages 1–6) plus cross-cutting concerns  
> **Goal:** Capture all known bugs, omissions, and “nice-to-haves,” then propose concrete fixes or enhancements—organized for easy reference.

---

## 🔹 1. `stage1_data_collection.py`

1. **`datetime` import in `suppress(ImportError)`**

   - **Issue:** Failing any third-party import silences the `datetime` import.
   - **Fix:** Move `from datetime import datetime` _above_ or _outside_ the `suppress` block.

2. **Confusing `pathlib as Path` alias**

   - **Issue:** You wrote `import pathlib as Path`, then call `Path.Path(...)`.
   - **Fix:** Use either
     ```python
     from pathlib import Path
     ```
     or
     ```python
     import pathlib
     ```
     and adjust calls accordingly.

3. **Undefined `boto3` if import suppressed**

   - **Issue:** Attempting `boto3.client(...)` will `NameError` if `boto3` was never bound.
   - **Fix:** Wrap S3 logic in its own try/except or explicitly `if "boto3" not in globals(): raise ImportError(...)`.

4. **Duplicate-row logging but no dropping**

   - **Issue:** We warn about duplicates but never dedupe.
   - **Enhancement:** Add a `drop_duplicates` flag or a separate `dedupe()` method.

5. **Missing return paths in `_postprocess()`**

   - **Issue:** Unexpected JSON shapes may yield `None` or `ValueError` without clarity.
   - **Fix:** Explicitly catch and re-raise with context (e.g. “REST response missing `data.records`”).

6. **Great Expectations block not fully guarded**
   - **Issue:** Missing suite or GE directory will crash.
   - **Fix:** Surround the entire GE block in try/except and log “suite `<name>` not found → skipping validation.”

---

## 🔹 2. `stage2_imputation.py`

1. **Invalid logistic‐regression “likelihood”**

   - **Issue:** Using `lr.score` as log‐likelihood and inverting `XᵀX` for SE is statistically unsound.
   - **Fix:** Replace with a proper likelihood-ratio test (e.g. `statsmodels.discrete.discrete_model.Logit`) or a specialized missingness package.

2. **Fragile TVD computation**

   - **Issue:** `.loc[common]` can KeyError or miss categories that appear only after imputation.
   - **Fix:** Compute TVD over the _union_ of categories, filling missing frequencies with 0.

3. **Unilateral column drop at 90 % NA**

   - **Issue:** You drop any column > 0.9 missing with no alternative strategy.
   - **Fix:** Expose a `drop_hi_na: bool` flag or implement a fallback imputation (e.g. MICE) for critical features.

4. **KNN imputer can be intractable at scale**

   - **Issue:** `KNNImputer` on large `n×p` blocks is slow.
   - **Enhancement:** If `n_rows > N` or `p > P`, either downsample or skip to median impute.

5. **Premature column dropping**

   - **Issue:** Dropping before feature-importance analysis may remove valuable predictors.
   - **Enhancement:** Delay hard drops until after a feature‐importance check (e.g. mutual information).

6. **Non-reproducible “random-sample” impute**

   - **Issue:** Calls to `np.random.choice` lack a `random_state`.
   - **Fix:** Use a `RandomState(self.random_state)` instance for reproducibility.

7. **Potential silent misalignment in `transform()`**
   - **Issue:** Relying on dropping `self.cols_to_drop` via names may silently skip if names change.
   - **Enhancement:** Assert that all expected `cols_to_drop` are indeed dropped (or warn if missing).

---

## 🔹 3. `stage3_outlier_detection.py`

1. **Over-eager winsorization**

   - **Issue:** Winsorizes any row with _any_ univariate vote, instead of “real” outliers.
   - **Fix:** Only cap rows with votes ≥ `multi_vote_threshold` (or expose a `winsorize_min_votes` parameter).

2. **Mahalanobis drops rows with any NA**

   - **Issue:** `df.dropna()` before covariance fitting may discard most of wide tables.
   - **Fix:** Impute missing or use pairwise‐complete Mahalanobis (e.g. based on robust covariance).

3. **Misaligned indices in `_modz_outliers()`**

   - **Issue:** Building `modz` on `arr = series.dropna().values` then zipping with `series.dropna().index` can misalign if the original `series` has gaps.
   - **Fix:** Compute directly on full series and mask NaNs:
     ```python
     modz = 0.6745 * (series - med) / mad
     return series[modz.abs() > cutoff].index.tolist()
     ```

4. **No “drop” option for outliers**

   - **Issue:** The class docs mention `drop_outliers=True` but no such argument exists.
   - **Fix:** Add `treatment="winsorize"|"drop"` in `__init__`.

5. **Report JSON lacking per-row votes**

   - **Issue:** We only write counts and a list of final indices.
   - **Enhancement:** Add a `"votes_per_row": { idx: vote_count, … }` section to the report.

6. **Blind reliance on χ² threshold for non-normal data**
   - **Issue:** If features aren’t Gaussian, Mahalanobis ppf cutoffs are meaningless.
   - **Enhancement:** Log the average kurtosis and warn if > 5 (or fallback to IF).

---

## 🔹 4. `stage4_scaling_transformation.py`

1. **Box-Cox λ not persisted**

   - **Issue:** You fit Box-Cox in train but cannot re-apply the same λ in `transform()`.
   - **Fix:** Store `self.boxcox_lambdas[col] = λ` and call `stats.boxcox(x, lmbda=λ)` on new data.

2. **Non-deterministic Shapiro on large arrays**

   - **Issue:** Subsampling via `np.random.choice` without a seed makes p-values non-reproducible.
   - **Fix:** Use `rng = RandomState(self.random_state)` for any subsampling.

3. **Re-fitting transforms on test set**

   - **Issue:** In `transform()`, Box-Cox, Yeo, and QT are re-fitted to test data.
   - **Fix:** Always apply the _fitted_ `PowerTransformer`/`QuantileTransformer` or persisted λ, never re-fit.

4. **Misplaced PCA code**

   - **Issue:** PCA logic appears under scaling module.
   - **Fix:** Remove from this file; it belongs in `stage6_pca.py`.

5. **No constant/near-zero-var guard**

   - **Enhancement:** If `std < ε`, skip all transforms on that column.

6. **Report JSON missing “why” for scaler choice**
   - **Enhancement:** In the JSON, include the exact skew/kurtosis values that triggered the chosen scaler.

---

## 🔹 5. `stage5_encoding.py`

1. **Unimplemented `transform()`**

   - **Issue:** No way to encode new data consistently.
   - **Fix:** Mirror the one-hot, ordinal, or target encodings from `fit_transform()` in a `transform()` method, preserving categories.

2. **Returns only “linear” variant**

   - **Issue:** Functions build `df_lin`, `df_tree`, `df_knn` but only return `df_lin`.
   - **Fix:** Return all variants as a tuple or allow user to select which one they want back.

3. **Arbitrary column ordering**

   - **Enhancement:** After concatenation, explicitly reorder columns (e.g. `df = df[sorted(df.columns)]`).

4. **No “RARE” binning before freq-encode**

   - **Enhancement:** Collapse categories under a `min_freq` threshold into a `"__RARE__"` bucket to stabilize encoding.

5. **No leakage guard in target encoding**
   - **Enhancement:** If you implement mean/WOE encoding, always do cross-validated encoding (e.g. with smoothing) to avoid target leakage.

---

## 🔹 6. `stage6_pca.py`

1. **Missing scaler persistence**

   - **Issue:** You fit a `StandardScaler` but never store it, then attempt to standardize via `pca_model.mean_`.
   - **Fix:** Store `self.scaler = StandardScaler().fit(X)` and use it in `transform()`.

2. **Condition number on raw, unscaled data**

   - **Enhancement:** Compute `cond()` on the _standardized_ covariance to detect numerical instability properly.

3. **No guard for `p < 2`**

   - **Issue:** Running PCA on 1 feature is meaningless.
   - **Fix:** `if len(self.numeric_cols) < 2: skip PCA`.

4. **No handling of zero-variance features**

   - **Enhancement:** Drop `df[col].std() == 0` before PCA.

5. **Inconsistent use of `PCA.mean_` vs. scale factors**
   - **Issue:** The division by `sqrt(explained_variance_)` is incorrect for standardization.
   - **Fix:** Always standardize via the stored `StandardScaler`, then project with the fitted PCA.

---

## 🔹 Cross-Cutting & Master-Pipeline

1. **Reproducibility via `random_state` everywhere**

   - Ensure _all_ classes accept a seed and use it for any random or subsampling.

2. **Centralized thresholds / config**

   - Move all numeric literals (`0.975`, `1.5×IQR`, `5000 rows`, `0.05 missing`, `500-level cutoff`) into a single `config.py` or `config.yaml`.

3. **Timestamped & versioned report files**

   - Append `_YYYYMMDD_HHMMSS` or a git commit hash to JSON outputs so you don’t overwrite previous runs.

4. **Unified logging setup**

   - In your “main” or notebook, call `logging.basicConfig(level=DEBUG, ...)` so that each module’s `log = logging.getLogger("stageX")` actually emits output.

5. **Master‐Pipeline wrapper**

   - Define a single class or function that orchestrates Stage 1→Stage 6 in order, persists each fitted transformer, and provides a single `preprocess(df_raw)` entrypoint.

6. **Unit tests for corner cases**

   - Write `pytest` cases for:
     - No numeric columns
     - All‐missing columns
     - Single‐row / single-col DataFrames
     - Tiny vs huge datasets (to catch skipped-at-scale logic)

7. **Flexible I/O formats**

   - Allow all stages to read/write CSV, Parquet, or Feather via a parameter (e.g. `engine="pyarrow"`).

8. **Early exit for supervised vs unsupervised modes**

   - Let missingness‐analysis, outlier‐detection, and encoding know whether a target exists and behave accordingly.

9. **Performance guards & profiling**

   - Before KNN, Mahalanobis, Box-Cox loops, check `n_rows * p > threshold` and either subsample or skip.

10. **Explicit mixed-type sniffing**
    - In a pre-step, find columns whose values mix strings/numerics and either coerce or isolate them for manual cleaning.

---

> _Implementing the **top 10 urgent fixes** (boxed in each section) will resolve the most critical correctness issues; the remaining 40+ “nice-to-haves” will round out robustness, reproducibility, and performance at scale._
````

**Next Steps:**

1. Paste this `improvements.md` into your repo.
2. Tackle the _boxed_ items in priority order.
3. Add unit tests or schema guards for each fix.
4. Wire everything into a single orchestrator for end-to-end reproducibility.

Good luck!

---

Below is a single, end-to-end **README.md** you can drop into your repo (or rename to `instruments.md`) that merges:

1. **Stanford/MIT/Harvard best-practice** life-cycle frameworks (CRISP-DM, TDSP, “classic ML flow”)
2. All of the dozens of concrete checks and improvements we discussed above
3. A clear, stage-by-stage checklist you can refer to for _every_ new project

Feel free to tweak thresholds or expand any section with your own notes.

---

````markdown
# Machine Learning Pipeline Reference

> **Purpose:**  
> This document codifies a reproducible, configurable, end-to-end ML pipeline from raw data ingestion through feature engineering — pulling together best practices from Stanford, MIT, Harvard and our in-house improvements.
>
> **How to use:**  
> For each project, walk through the numbered stages below. Each stage lists **Objectives**, **Key Tasks**, **Checks & Metrics**, and **Configurable Parameters**.

---

## Table of Contents

1. [Stage 0: Project Setup & Configuration](#stage-0-project-setup--configuration)
2. [Stage 1: Data Ingestion](#stage-1-data-ingestion)
3. [Stage 2: Data Validation & Sanity Checks](#stage-2-data-validation--sanity-checks)
4. [Stage 3: Missingness & Cleaning](#stage-3-missingness--cleaning)
5. [Stage 4: Outlier Detection & Treatment](#stage-4-outlier-detection--treatment)
6. [Stage 5: Scaling & Transformation](#stage-5-scaling--transformation)
7. [Stage 6: Feature Engineering & Selection](#stage-6-feature-engineering--selection)
8. [Stage 7: Train/Val/Test Split & Baseline](#stage-7-trainvaltest-split--baseline)
9. [Stage 8: Leakage Detection & Integrity](#stage-8-leakage-detection--integrity)
10. [Next Steps](#next-steps)

---

## Stage 0: Project Setup & Configuration

### Objectives

- Centralize all thresholds, file paths, random seeds, reporting options.
- Ensure reproducibility (version your code, record git SHA).

### Key Tasks

- Create `config.yaml` (or Python dict) with:
  ```yaml
  data:
    raw_path: data/raw/
    proc_path: data/processed/
  thresholds:
    high_cardinality_ratio: 0.10
    missing_frac_drop: 0.90
    outlier_vote_threshold: 3
    nzv_unique_threshold: 2
    skew_robust: 1.0
    kurtosis_robust: 5.0
    shapiro_p_thresh: 0.05
  random_seed: 42
  ```
````

- Record experiment metadata: git commit, timestamp, config hash.

### Checks & Metrics

- Validate `config.yaml` schema (e.g. with [Cerberus](https://docs.python-cerberus.org/)).
- Halt if any required key is missing.

---

## Stage 1: Data Ingestion

### Objectives

- Load raw tables (CSV, Parquet, database) into a canonical in-memory format.
- Persist a snapshot (Parquet) for downstream stages.

### Key Tasks

1. **Read source** into `pd.DataFrame` (or Spark/DB connector).
2. **Schema enforcement**:

   - Define expected dtypes & nullability in a schema file (JSON/Avro/tdl).
   - Cast or reject mismatches.

3. **Write** cleaned snapshot to `data/processed/data.parquet`.

### Checks & Metrics

- Row counts vs. source logs.
- Column count & dtypes match schema.
- Checksum Parquet file for integrity.

---

## Stage 2: Data Validation & Sanity Checks

> _Inspired by Harvard’s “Data Quality Framework”._

### Objectives

- Detect structural, semantic, and cross-field anomalies before heavy processing.

### Key Tasks & Checks

1. **Mixed-Type Detection**

   - For each column:

     ```python
     types = df[col].dropna().map(type).value_counts()
     if len(types) > 1: report.mixed_types[col] = types.to_dict()
     ```

2. **Impossible Value Rules**

   - User-supplied forbidden sets (e.g. negative ages, zeros in IDs).

3. **Unexpected High-Cardinality**

   - `unique_ratio = nunique / n_rows`; if `> config.thresholds.high_cardinality_ratio`, flag.

4. **Near-Zero-Variance**

   - Drop if `nunique < config.thresholds.nzv_unique_threshold`.

5. **Duplicate Rows**

   - `df.duplicated().sum()`. Optionally remove.

6. **Custom Missing Markers**

   - Scan short strings (`len<=2`) with `freq > 1%` → treat as null.

---

## Stage 3: Missingness & Cleaning

> _Leverage MIT’s “Robust Missing Data” guidelines._

### Objectives

- Characterize missingness mechanism & apply robust imputation or flagging.

### Key Tasks

1. **Missingness Pattern Analysis**

   - Little’s omnibus MCAR test & per-column logistic test.

2. **Stratified Missingness**

   - `df.groupby(target)[col].apply(lambda s: s.isna().mean())`.

3. **Drop vs Impute**

   - Drop if `missing_frac > config.thresholds.missing_frac_drop`.
   - Else impute—choose strategy per column:

     - Numeric: mean/median/KNN (only if `n_rows <= knn_max_rows`) /random.
     - Categorical: mode/constant/random via TVD.

4. **Cast Mixed-Type**

   - If strings in numeric column ≥90% digit-like, cast to numeric.

### Checks & Metrics

- Report per-column missing fractions & chosen method.
- Report covariate shift pre/post imputation.

---

## Stage 4: Outlier Detection & Treatment

### Objectives

- Detect both univariate & multivariate outliers, then either cap (winsorize) or drop.

### Key Tasks

1. **Univariate Rules** (each gives 1 vote)

   - IQR (1.5×), Z-score (|z|>3), ModZ (|modz|>3.5), Tukey (2× IQR), 1st/99th pct.

2. **Multivariate Rules** (row-complete only)

   - If `n_rows >= 5×n_features` & ≥60% Gaussian: Mahalanobis (χ² 97.5%).
   - Else if `n_rows<2k & n_features<50`: LOF (novelty).
   - Else: IsolationForest (contamination 0.01).
   - Special case: if `n_features ≥ n_rows`, skip directly to IsolationForest.

3. **Voting & Threshold**

   - Flag real outliers if votes ≥ `config.thresholds.outlier_vote_threshold`.

4. **Treatment**

   - If `cap_outliers=True`: Winsorize at each column’s 1st/99th pct.
   - Else: drop rows.

### Checks & Metrics

- Number & fraction of rows flagged by each method.
- Final outlier count & treatment summary.

---

## Stage 5: Scaling & Transformation

> _Based on MIT’s “Adaptive Scaling” paper._

### Objectives

- Drop near-zero-variance, choose an optimal scaler, then apply extra transforms to nudge toward normality.

### Key Tasks

1. **Drop NZV Columns** (`nunique < config.nzv_threshold`).
2. **Choose Scaler** by skew/kurtosis:

   - If any `|skew| > sk_thresh_robust` or `|kurt| > kurt_thresh_robust` → RobustScaler
   - Else if ALL `|skew| < sk_thresh_standard` → StandardScaler
   - Else → MinMaxScaler

3. **Fit & Apply Scaler** → `self.scaler`.
4. **Post-Scale Normality Check** (Shapiro p > shapiro_p_thresh & |skew|\<transform_skew_thresh).
5. **Extra Transforms** (if not “Gaussian enough”):

   - Trial: none, Box-Cox (if all >0), Yeo-Johnson, Quantile→Normal
   - Choose by lexicographic `(pval, −|skew|)`.

6. **Persist** `scaler`, per-col transformer for `transform()`.

### Checks & Metrics

- Report chosen scaler & per-column transform with scores.
- JSON or DataFrame–style report for drift monitoring.

---

## Stage 6: Feature Engineering & Selection

> _Following Stanford’s “Feature Lab” guidelines._

### Objectives

- Add generic derived features, prune redundancy, preselect for modeling.

### Key Tasks

1. **Interaction Candidates**

   - Optionally generate pairwise (`A×B`, `A+B`) for top N numeric pairs by correlation.

2. **Binning / Grouping**

   - Auto-bin numeric into quartiles or quantile buckets.

3. **Categorical Encoding**

   - Target or frequency encoding for high-cardinality (>10 levels).

4. **Correlated Feature Pruning**

   - Drop one of any pair with `|corr| > config.thresholds.drop_corr` (e.g. 0.95).

5. **Recursive Feature Elimination (RFE)**

   - Light wrapper using your chosen estimator to preselect top K features.

6. **Save** transformation summary for `transform()`.

### Checks & Metrics

- Final feature count.
- Variance Inflation Factor (VIF) to detect multicollinearity.
- Permutation‐importance snapshot.

---

## Stage 7: Train/Val/Test Split & Baseline

> _CRISP-DM Step: Modeling, but here only splitting & baseline._

### Objectives

- Create reproducible splits; optionally oversample; compute trivial baselines.

### Key Tasks

1. **Parquet Splits**

   - Train/Val/Test: 60/20/20 stratified on target if requested.

2. **Oversampling**

   - If classification (`y.dtype.kind in {'O','i','b'}`) & `oversample=True`, apply SMOTE on train only.

3. **Baseline Models**

   - Regression: mean predictor → MAE, R²
   - Classification: majority class → accuracy, F1
   - (Optionally add k-NN vs naive Bayes as extra baselines.)

4. **Sanity Checks**

   - No index overlap between splits.
   - No feature identical to target.

### Checks & Metrics

- Row counts per split.
- Baseline metric summary in JSON.

---

## Stage 8: Leakage Detection & Integrity

### Objectives

- Catch both target-leakage and train/test separation leakage.

### Key Tasks

1. **Target-Leakage**

   - For each feature: compute AUC( feature → target ) > auc_thresh (e.g. 0.99)
   - Categorical: one-hot & take max AUC.

2. **Train/Test Separation**

   - Concat train/test, label 0/1, compute AUC( feature → is_train ).

3. **Row Overlap**

   - Merge on all features – if any exact duplicate, warn.

4. **Unseen Categories**

   - For each cat col: test_vals – train_vals → report.

### Checks & Metrics

- List of leaky features.
- Separation AUC per feature.
- Overlap row count.

---

## Next Steps & MLOps

1. **Model Selection & Tuning**

   - Wrap with [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/pipeline.html), grid/RandomSearchCV.

2. **Experiment Tracking**

   - Log metrics/params to MLflow or Weights & Biases.

3. **Model Validation**

   - Cross-validation, calibration curves, partial dependence.

4. **Deployment**

   - Package as API (FastAPI), containerize (Docker), auto-retrain triggers.

5. **Monitoring**

   - Periodic data-drift & prediction-drift checks, alerting on threshold breaches.

---

> **References & Further Reading**
>
> - CRISP-DM (SPC Whitepaper)
> - TDSP (Microsoft’s Team Data Science Process)
> - “Feature Lab” at Stanford CS229
> - MIT 6.86x “Machine Learning on Big Data” lecture notes
> - Harvard Data Science Common Workflow

_(This README is meant as a living document – feel free to iterate over it as your team’s needs evolve.)_
