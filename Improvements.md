# Improvements

## 1. Phase 3 – Data Preparation

<!-- Great Expectations or pandera for schema assertions
After loading, run a minimal GE check or pandera.DataFrameSchema to assert column types, ranges, uniqueness of ID, etc.

Example: ensure that primary key is unique, order_date parses to a datetime.

If any check fails, log and exit with nonzero status. -->

1. **3B1: De‑duplication & Invariant Pruning (Missing)**
   You currently jump straight to missing‐value imputation. But many production pipelines:

   - **Drop 100%‐constant columns** (zero variance).
   - **Drop columns with > X% missing** (e.g. > 80%).
   - **Drop duplicate rows** (exact duplicates).
     The supplied `DataPreparer` does **not** perform de‑duplication or “prune invariant features.” Add a sub‐step:

2. **3D: Expand “Data Transformation & Scaling” options**

   - Currently you log‐transform only `amount`. But you may need “box‐cox” on other positive features (e.g. `income`).
   - The code path for `--scaler yeo` already builds `PowerTransformer(method='yeo‑johnson')`, but it’s only applied on all numeric columns—even those already close to normal. You could let users pass something like:

     ```bash
     python -m ml_pipeline.prepare --scaler standard --power blah --log_cols amount,income
     ```

     so it’s easier to specify which columns to “power‐transform” vs. just scale.

3. **IsolationForest contamination parameter**

   - In `treat_outliers`, you hardcode `contamination=0.01`.
   - Better to expose it as `--contamination 0.02` or similar, so that very tiny or moderate outliers can be controlled at runtime.

4. **3F: Data Versioning & “raw_sha”**

   - In `save()`, you write `"raw_sha": self.cfg.get("raw_sha","n/a")`. But where does `raw_sha` come from?
   - If you want to tie lineage from Phase 2 → Phase 3, you could pull in the checksum that `OmniCollector` wrote into `logs/ingest.log` and embed that here. If you never set `raw_sha` in the CLI, it remains `"n/a"`, which is deceptive.
   - Suggestion: When you run `python -m ml_pipeline.prepare`, add a `--raw-sha` CLI flag so that the user (or a shell wrapper) can pass the SHA it just ingested.

5. **Add data quality reporting in Phase 3**

   - Generate a simple report (CSV or JSON) listing:

     - Number of nulls per column before/after imputation
     - Number of outliers removed
     - Distribution of numeric columns (mean/median/percentiles)
       That can go under `reports/lineage/data_preparation_report.json` for audit purposes.

6. **Split “clean” vs “scaled”**
   Right now `self.df.to_parquet(INT)` and then immediately `self.df.to_parquet(PROC)`. As a result, both `interim/clean.parquet` and `processed/scaled.parquet` contain exactly the same file. Usually you expect:

   - `data/interim/clean.parquet` = before scaling & imputation
   - `data/processed/scaled.parquet` = after scaling & imputation
     But the current code applies imputation and scaling **in‐memory** then writes the same DataFrame twice. If you intended “interim” 👉 “no scaling” and “processed” 👉 “scaled,” you should do:

   ```python
   clean_df = self.df.copy()
   clean_df.to_parquet(INT, index=False)
   # then scale/impute on clean_df to get proc_df
   proc_df = …
   proc_df.to_parquet(PROC, index=False)
   ```

   Otherwise you have no record of what the data looked like pre‐scale.

---

3. **Profiles: Check if `ydata-profiling` import fails**
   In `EDA.py` you do:

   ```python
   try:
       from ydata_profiling import ProfileReport
       PROFILING_OK = True
   except ModuleNotFoundError:
       PROFILING_OK = False
   ```

   But if the user calls `python -m Data_Analysis.EDA --profile`, they get “install ydata_profiling” and return.

   - Suggestion: Raise a user‐friendly message (“pip install -e .\[eda]”) so folks know how to get that extra dependency.
   - Also, verify at the top of `pyproject.toml` you truly have `extras = { eda = ["ydata-profiling>=4.6"] }`.

4. **EDA order of operations**

   - You currently do univariate (4A), then bivariate (4B), then multivariate (4C).
   - In many workflows, it’s wise to do **Feature Selection** (e.g. drop near‐zero‐var columns, check for multicollinearity) before you do your “full” EDA so you don’t waste time plotting hundreds of useless variables.
   - Either add a Phase 4.5 (as you have in the README) explicitly in code, or modify `EDA.py` to accept `--early-filter` which prunes useless columns before plotting.

5. **“Leakage sniff” enhancement**

   - You only check numeric columns that correlate (AUC > 0.97) with the target. What about categorical‐valued “DateTime” columns (timestamp that always comes after target)?
   - You could add a step to compute “for each datetime column, is there any record where `timestamp > label_date`? If so, flag”
   - Also, if you detect any text column whose TF‑IDF dimension has extremely high information (e.g. a single token has near‐perfect separation), warn the user.

6. **Automate saving a master HTML report**

   - Right now you save `reports/eda/profile.html` but you only do that if `--profile`.
   - If the project wants a “one‐click EDA,” consider making `--mode all` default to also trigger `--profile`.
   - Provide a symlink or index page so that once you run `python -m Data_Analysis.EDA --mode all --profile`, you can open a single HTML file with all plots embedded (e.g. in `reports/eda/index.html`).

7. **Use vectorized I/O instead of saving dozens of PNGs**
   If your dataset has hundreds of numeric columns, your current EDA pipeline will generate one PNG per column. That can be painful to navigate. You might:

   - **Group similar columns into sub‐folders** (`reports/eda/uva/plots/numerics/`, `…/categoricals/`).
   - Use multi‐page PDF or HTML figure so the user can scroll through all univariate plots in one place (e.g. `reportlab` or a simple Jupyter notebook that loops through all columns).
   - Allow a `--max-cols 20` flag so you only produce the “top 20 variables” by variance or by missingness, etc.

---

## 2. Phase 4½ – Feature Selection & Early Split

_Currently this is only mentioned in the README; there is no “actual code” under `src/Data Analysis` for Phase 4.5._ If you truly want an early feature‐selection step (prune near‐zero variance, drop > X% missing, drop high‐correlation pairs) **before** running Feature Engineering, you need:

1. **A new script, e.g.**
   `src/data_analysis/feature_selection.py` that implements:

   - **Zero/near‐zero variance** (`VarianceThreshold`)
   - **High‐missingness filter** (`df.isna().mean() > 0.5`)
   - **High‐null correlation** (if `corr(x,y)>0.95` drop `y`)
   - **Mutual information filter** (bottom 10% of MI w/ target)
   - Possibly **chi‑square filter** for categorical features.
     At the end, produce:

   ```json
   {
     "dropped_columns": ["colA","colB","colC"],
     "kept_columns": ["colX","colY",…],
     "mi_scores": { "colX":0.23, "colY":0.11, … }
   }
   ```

   and write that to `reports/feature/feature_audit.json`.

2. **Hook early splitting here (if you want)**
   If you want to do `train/val/test` earlier (instead of Phase 5.5), call

   ```bash
   python -m data_analysis.feature_selection --data data/interim/clean.parquet --target is_churn \
       --split-ratio 0.8,0.1,0.1 --stratify \
       --out-splits data/splits/train.parquet,data/splits/val.parquet,data/splits/test.parquet
   ```

   and then save `data/splits/split_manifest.json`.

3. **Update README to explain “why do we split here vs. Phase 5.5?”**
   Right now the README shows Phase 5.5 as the canonical split. If you do an earlier split, update the doc so that downstream phases only use training data for feature engineering.

---

## 3. Phase 5 – Feature Engineering

Your existing `feature_engineering.py` (and the “v3” version in your notes) are already very powerful. Below are items to tighten:

1. **Unused parameters**
   In the “v3” class definition you have:

   - **Problem:** `interactions` is never used in `_build_pretransform` or `_build_column_transformer`.
   - **Fix:** Implement `if self.interactions: n_pipe.append(("interact", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))` or similar, or remove the parameter entirely if you don’t plan to support polynomial interactions.

2. **Datetime expansion class missing/unreferenced**

   - In version v2 you refer to a class `DatetimeExpand(drop=False)`, but there is no such class in that file.
   - **Action:** Either implement a small “DatetimeExpand” transformer that splits a `Timestamp` into year/month/day/dayofweek/hour/minute, OR remove references to it and document in README that “for datetime expansion, please add your own `DatetimeExpand` plug‑in.”

3. **Column transformer “remainder='drop'” vs “passthrough”**

   - Currently you drop any column that isn’t numeric, categorical or text. But what if you have a “latitude” or “order_id” that you want to leave alone (maybe as an ID you drop later)?
   - Make it clear in README: “We drop any column not specifically listed under `text_cols`, `num`, or `cat`. If you have “identifier” columns, explicitly drop them in a custom step or earlier in Phase 3.”

4. **Aggregation features (group‐by) are not implemented**
   You accept a parameter `aggregations: Dict[str,List[str]]` (e.g. `{"cust_id": ["amt_mean","amt_sum"]}`), but nowhere is that applied.

   - **Fix:** In `_build_pretransform`, after filtering, add something like:

     ```python
     if self.aggregations:
         for grp_col, agg_list in self.aggregations.items():
             for func in agg_list:
                 col, agg = func.split("_",1)   # e.g. “amt_mean” → (“amt”, “mean”)
                 new_name = f"{grp_col}_{col}_{agg}"
                 X[new_name] = X.groupby(grp_col)[col].transform(agg)
     ```

   - Document in README how to specify these.

5. **“RareCategory” threshold units**

   - You treat `th` as a float fraction, but your constructor doc says “th: float | int = .01 # 1 % or absolute count.” Then you check `if isinstance(self.th, float) else vc < self.th/len(X)`. That is a little confusing.
   - **Better:** Let `rare_threshold` always be “proportion” (0.01 = 1 %) and remove integer alternative. If you really want integer thresholds, clarify in docstring and rename parameter to something like `rare_threshold_frac` and `rare_threshold_abs`.

6. **Feature‐audit JSON needs more detail**
   Currently you write:

   ```python
   audit = {
     "n_features_in": len(X.columns),
     "n_features_after_clean": len(X_clean.columns)
   }
   (self.report_dir/"feature_audit.json").write_text(json.dumps(audit, indent=2))
   ```

   But that only tells users how many features remain, not which features were dropped. A better “audit” would look like:

   ```json
   {
     "n_features_in": 50,
     "dropped_nzv": ["colA", "colB"],
     "dropped_corr": ["colC"],
     "dropped_low_mi": ["colD"],
     "kept_features": ["colE","colF",…]
   }
   ```

7. **After fit, write out a “preprocessor_manifest.json”**
   In addition to dumping `preprocessor.joblib`, also write:

   ```json
   {
     "pipeline_sha256": "aba1e5c…",
     "config": {
       "numeric_scaler": "standard",
       "numeric_power": "yeo",
       "log_cols": ["amount"],
       "quantile_bins": {"age":4},
       …
     },
     "timestamp": "2025-05-31T20:45:03Z"
   }
   ```

   This lets you trace exactly which config was used to build the transform.

8. **Add unit tests for every custom transformer**
   For instance, in `tests/test_feature_engineering.py`:

These tests catch “silent failure” if a transformer parameter is mis‐typed.

---

## 4. Phase 5½ – Dataset Partition & Baseline Benchmarking

1. **Baseline metrics for regression + classification in one place**

   - Today if `y_test` is continuous, you compute MAE + R²; if categorical, you compute accuracy + F1.
   - Add other metrics for classification (ROC‐AUC, PR‐AUC) and for regression (RMSE, MAPE). Even if you don’t record them, log them at least to console or JSON so future users see them.

2. **Train/Test “Sanity Checks”**
   Right now you only check:

   - No duplicate indices
   - No column that exactly duplicates the target
     But you should also check:
   - **No feature means in test that are 10× greater than train** (distribution shift sniff)
   - **Categorical levels in test must be a subset of train** (no unseen categories, or if an unseen category appears, you’ll get errors in one‐hot encoding unless you used `handle_unknown='ignore'`).
   - Update the code to do:

     ```python
     for col in train.columns:
         if col != self.target and train[col].dtype == object:
             unseen = set(test[col].unique()) - set(train[col].unique())
             assert not unseen, f"Test has unseen categories in {col}: {unseen}"
     ```

3. **Freeze preprocessor more generically**
   You currently rebuild a brand‑new `StandardScaler()` on `self.df` (full dataset) and dump that. But:

   - If your actual feature engineering pipeline was a custom `ColumnTransformer` from Phase 5 (e.g. you used `StandardScaler` only on “age” and “amount”), you should freeze that same pipeline, not build a new one from scratch.
   - **Fix:** In `freeze_preprocessor`, instead of creating a fresh `Pipeline([("scale", StandardScaler())])`, load the pipeline you already saved in Phase 5 (`models/preprocessor.joblib`) and simply validate that all numeric features are scaled. Then re‐dump it (or confirm its SHA) so it matches.

## 8. Code Quality / Style

1. **Run a linter prettier with strict rules**

2. **Remove bare `except Exception` / suppress warnings globally**

   - In many places you catch a broad `ModuleNotFoundError` and set a boolean. Instead, catch precisely what you need (e.g. `except ImportError:`). Otherwise you might hide a real bug.
   - For logging, avoid using bare `print()` statements in library code; instead use `log.info()`, `log.error()`, etc.

3. **Split large files into smaller modules**

   - `feature_engineering.py` (and `data_preparation.py`) are \~200–300 LOC each. You could break them into:

     ```
     src/Feature_Engineering/
       ├── __init__.py
       ├── transformers.py      # FrequencyEncoder, RareCategory, TextLength, Cyclical, etc.
       ├── selection.py        # VarianceThreshold, MI filter, correlation filter
       └── feature_engineering.py  # orchestrates ColumnTransformer + pipeline
     ```

   - That makes it easier to test each module independently (e.g. `tests/test_transformers.py`, `tests/test_selection.py`, `tests/test_pipeline.py`).

4. **Logging vs. print**
   Everywhere you currently do `print("…")` (e.g. in `split_and_baseline.sanity_checks`), switch to `log.info("…")` or `log.error("…")`. That way you can control verbosity with `logging.basicConfig(level=logging.INFO)`.

---

## 9. Final Touches

1. **Consistency between `FeatureEngineer` constructor parameter names and README**

2. **Raise informative errors when mandatory parameters are missing**

### Recap of Major Improvement Areas

- **Code Quality:**

  1. Lint with `flake8`/`black`/`mypy`.
  2. Remove unused parameters (e.g. `interactions`, `self.text_params`).
  3. Add missing pieces (DatetimeExpand, aggregations, interactions).
  4. Fix “oversample” condition, stratified splitting bugs.
  5. Add more granular EDA pipeline (Phase 4.5), share code between EDA scripts.

## 10. Example flow with Synthetic Dataset
