# Improvements

## 1. Phase 3 – Data Preparation

Great Expectations or pandera for schema assertions

After loading, run a minimal GE check or pandera.DataFrameSchema to assert column types, ranges, uniqueness of ID, etc.

Example: ensure that primary key is unique, order_date parses to a datetime.

If any check fails, log and exit with nonzero status.

1. **3B1: De‑duplication & Invariant Pruning (Missing)**
   You currently jump straight to missing‐value imputation. But many production pipelines:

   - **Drop 100%‐constant columns** (zero variance).
   - **Drop columns with > X% missing** (e.g. > 80%).
   - **Drop duplicate rows** (exact duplicates).
     The supplied `DataPreparer` does **not** perform de‑duplication or “prune invariant features.” Add a sub‐step:

   ```python
   def drop_constant(self):
       nunique = self.df.nunique()
       const_cols = nunique[nunique <= 1].index.tolist()
       self.df.drop(columns=const_cols, inplace=True)
       log.info("Dropped %d constant columns", len(const_cols))
   ```

   and call it at the start of `run()`.

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

   ```python
   def __init__(
       target: str | None = None,
       numeric_scaler: str = "standard",
       numeric_power: str | None = None,
       log_cols: List[str] | None = None,
       quantile_bins: Dict[str,int] | None = None,
       polynomial_degree: int | None = None,
       interactions: bool = False,           # ⚠ UNIMPLEMENTED
       rare_threshold: float | None = None,
       cat_encoding: str = "onehot",
       text_vectorizer: str | None = None,
       text_cols: List[str] | None = None,
       datetime_cols: List[str] | None = None,
       cyclical_cols: Dict[str,int] | None = None,
       date_delta_cols: Dict[str,str] | None = None,
       aggregations: Dict[str,List[str]] | None = None,
       # NEW —— filtering / pruning
       drop_nzv: bool = True,
       corr_threshold: float | None = .95,
       mi_quantile: float | None = .10,
       # misc
       custom_steps: List[Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
       save_path: str | Path = "models/preprocessor.joblib",
       report_dir: str | Path = "reports/feature"
   ):
       self.__dict__.update(**locals())
       ...
   ```

   - **Problem:** `interactions` is never used in `_build_pretransform` or `_build_column_transformer`.
   - **Fix:** Implement `if self.interactions: n_pipe.append(("interact", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))` or similar, or remove the parameter entirely if you don’t plan to support polynomial interactions.

2. **Datetime expansion class missing/unreferenced**

   - In version v2 you refer to a class `DatetimeExpand(drop=False)`, but there is no such class in that file.
   - **Action:** Either implement a small “DatetimeExpand” transformer that splits a `Timestamp` into year/month/day/dayofweek/hour/minute, OR remove references to it and document in README that “for datetime expansion, please add your own `DatetimeExpand` plug‑in.”

3. **`text_params` is never defined**
   In your “v3” code (and v2) you do:

   ```python
   for tcol in txt:
       transformers.append(
           (f"text_{tcol}", text_map[self.text_vectorizer](**self.text_params), tcol))
   ```

   but there is no `self.text_params`. This will raise `AttributeError` at runtime.

   - **Fix:** Decide what hyperparameters you want for your text vectorizer (e.g. `max_features=100, stop_words='english'`) and define `self.text_params = {"max_features": 100, "ngram_range": (1,2)}` in `__init__`.
   - Or allow user to pass `text_params: dict` into the constructor.

4. **Column transformer “remainder='drop'” vs “passthrough”**

   - Currently you drop any column that isn’t numeric, categorical or text. But what if you have a “latitude” or “order_id” that you want to leave alone (maybe as an ID you drop later)?
   - Make it clear in README: “We drop any column not specifically listed under `text_cols`, `num`, or `cat`. If you have “identifier” columns, explicitly drop them in a custom step or earlier in Phase 3.”

5. **Aggregation features (group‐by) are not implemented**
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

6. **“RareCategory” threshold units**

   - You treat `th` as a float fraction, but your constructor doc says “th: float | int = .01 # 1 % or absolute count.” Then you check `if isinstance(self.th, float) else vc < self.th/len(X)`. That is a little confusing.
   - **Better:** Let `rare_threshold` always be “proportion” (0.01 = 1 %) and remove integer alternative. If you really want integer thresholds, clarify in docstring and rename parameter to something like `rare_threshold_frac` and `rare_threshold_abs`.

7. **Feature‐audit JSON needs more detail**
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

   That way, the downstream “feature dictionary” generator can show exactly what was pruned.

8. **Implement text‐length and “word‐count” as new numeric features**
   You already have a `TextLength` transformer in v3, but you only append it if `text_cols` is defined. Make sure you actually use it in the pipeline. For each `text_col`:

   ```python
   transformers.append((f"textlen_{col}", TextLength(), [col]))
   ```

   so that the two new columns (`*_n_chars`, `*_n_words`) show up as numeric arrays.

9. **After fit, write out a “preprocessor_manifest.json”**
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

10. **Add unit tests for every custom transformer**
    For instance, in `tests/test_feature_engineering.py`:

```python
import pandas as pd
import numpy as np
from Feature_Engineering.feature_engineering import RareCategory, TextLength

def test_rare_category_float_threshold():
    df = pd.DataFrame({"A": ["x","x","y","z"]})
    rc = RareCategory(th=0.5).fit(df)
    arr = rc.transform(df)
    # “z” should be replaced by "__rare__"
    assert "__rare__" in df.iloc[arr[:,0] == "__rare__", 0].values

def test_text_length():
    df = pd.DataFrame({"comment": ["hello world",""]})
    tl = TextLength().fit(df)
    out = tl.transform(df)
    # First row: 11 chars, 2 words
    assert out[0,0] == 11
    assert out[0,1] == 2
```

These tests catch “silent failure” if a transformer parameter is mis‐typed.

---

## 4. Phase 5½ – Dataset Partition & Baseline Benchmarking

1. **Simplify the “oversample” logic**
   In `SplitAndBaseline.split_data` you check:

   ```python
   if self.oversample and y.dtype.kind not in "if":
       X_tr, y_tr = …
       X_tr, y_tr = SMOTE(…).fit_resample(X_tr, y_tr)
       train = pd.concat([X_tr, y_tr], axis=1)
   ```

   - **Issue:** `y.dtype.kind not in "if"` means “oversample only if `y` is _not_ numeric,” but that’s backwards. If it’s classification, `y.dtype.kind` would usually be `“i”` (integer), so `“i” not in “if”` is false → no oversample.
   - **Fix:** Flip that condition:

     ```python
     if self.oversample and y.dtype.kind in {"O","i","b"}:  # or simply presence of .nunique() <= 2
         …
     ```

   - Or better yet, explicitly require `--oversample` only for classification tasks and error out if used on continuous targets.

2. **Stratification consistency**

   But your “strat” for `temp` should be `strat.loc[temp.index]` (as you do in the first split), not `strat` itself. Otherwise, the second split may not be truly stratified by classes.

   - **Fix:**

     ```python
     strat_temp = strat.loc[temp.index] if self.stratify else None
     val, test = train_test_split(temp, …, stratify=strat_temp)
     ```

3. **Baseline metrics for regression + classification in one place**

   - Today if `y_test` is continuous, you compute MAE + R²; if categorical, you compute accuracy + F1.
   - Add other metrics for classification (ROC‐AUC, PR‐AUC) and for regression (RMSE, MAPE). Even if you don’t record them, log them at least to console or JSON so future users see them.

4. **Train/Test “Sanity Checks”**
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

5. **Freeze preprocessor more generically**
   You currently rebuild a brand‑new `StandardScaler()` on `self.df` (full dataset) and dump that. But:

   - If your actual feature engineering pipeline was a custom `ColumnTransformer` from Phase 5 (e.g. you used `StandardScaler` only on “age” and “amount”), you should freeze that same pipeline, not build a new one from scratch.
   - **Fix:** In `freeze_preprocessor`, instead of creating a fresh `Pipeline([("scale", StandardScaler())])`, load the pipeline you already saved in Phase 5 (`models/preprocessor.joblib`) and simply validate that all numeric features are scaled. Then re‐dump it (or confirm its SHA) so it matches.

6. **Write a “split_manifest.json” with checksums of each split file**
   You already write row counts and seed. Extend the JSON to include:

   ```json
   {
     "train_sha256": "<sha256-of data/splits/train.parquet>",
     "val_sha256": "<…>",
     "test_sha256": "<…>"
   }
   ```

   That way, if two users run `make baseline` on a fresh clone, they know they truly have the same splits.

---

## 5. Phase 6+ – Model Design & Training (Next Steps)

3. **“Early stopping” / regularisation logging**

   - For tree‐based models, add a callback to log feature importance & early stopping.
   - For logistic regression, allow both L1 and L2 so you can produce “sparse” weight vectors.

---

## 6. Phase 7+8+9 – (Placeholders or Next Steps)

1. **Evaluation & explainability (SHAP/LIME)**

   - Create a Jupyter notebook under `notebooks/` that demonstrates:

     - How to load a saved model and produce a SHAP summary plot.
     - How to produce a “model card” with summary metrics.

   - Add a subheading in the README: “Phase 7 – Model Explainability via SHAP.”

2. **Monitoring & drift detection**

   - If you want to show a minimal proof‑of‑concept (POC), add a script:

     ```bash
     python src/monitoring/drift_detector.py --live-data new_data.parquet --pretrained-model models/model.pkl
     ```

     that uses something like `evidently` or `scikit‑multiflow` to compute population drift.

3. **Deployment (Phase 8)**

   - Provide a Dockerfile in `docker/` that:

     - Installs `mldlc-toolkit`
     - Copies `models/model.pkl` and `models/preprocessor.joblib`
     - Sets up a small Flask/FastAPI service (`src/deploy/app.py`) that loads the preprocessor and model and exposes an endpoint `/predict`.

   - In the README, document how to run:

     ```bash
     docker build -t my‐model‐service .
     docker run -p 8080:8080 my-model-service
     curl -X POST http://localhost:8080/predict -d '{"features": […]}'
     ```

---

## 7. params.yaml

1. **Group keys by logical stage**
   Instead of scattering keys under “collect,” “split,” etc., you might create a top‐level structure:

   ```yaml
   phase2:
     collect_args: "file data/raw/users.csv --redact-pii --save"
   phase3:
     knn: false
     outlier: "iqr"
     scaler: "standard"
     balance: "none"
   phase4:
     eda_mode: "all"
     eda_target: "is_churn"
     eda_profile: true
   phase4_5:
     drop_nzv: true
     corr_threshold: 0.95
     mi_quantile: 0.1
   phase5:
     numeric_scaler: "standard"
     numeric_power: "yeo"
     log_cols: ["amount"]
   phase5_5:
     seed: 42
     stratify: true
     oversample: false
   target: "is_churn"
   ```

   This way, each section of `dvc.yaml` can refer to `params: ["phase5.numeric_scaler", "phase5.numeric_power", …]` etc.

---

## 8. README.md (current)

Below are items in your existing README that you should revisit:

3. **Phase 3 “3G Feature Pruning” is in TOC but not in content**

   - In your Table of Contents under “Phase 3 – Data Preparation,” you list a “3G Feature Pruning (High NaN / High Corr).” But in section 3 you never mention “3G.”
   - **Fix:** Either implement “3G” in `data_preparation.py` (e.g. drop features with `df.isna().mean() > 0.5` or `|corr(x,y)| > 0.95`), or remove `3G` from the TOC.

4. **Add a “How to run the pipeline start→finish” section**
   At the bottom, after your TOC, add:

   ````markdown
   ## How to run (one‐line)

   1. Install (dev):
      ```bash
      pip install -e .[eda,tests]
      ```
   ````

   2. Pull DVC data:

      ```bash
      dvc pull
      ```

   3. Prepare data:

      ```bash
      python -m ml_pipeline.prepare --knn --outlier=robust --scaler=standard --balance=smote
      ```

   4. Run EDA:

      ```bash
      python -m data_analysis.EDA --mode all --profile --target is_churn
      ```

   5. Feature Engineering:

      ```bash
      python -m Feature_Engineering.feature_engineering --data data/interim/clean.parquet --target is_churn \
          --numeric_scaler=robust --log_cols=amount --quantile_bins='{"age":4}'
      ```

   6. Split & Baseline:

      ```bash
      python -m Data_Cleaning.split_and_baseline --target is_churn --stratify --seed=42
      ```

   7. train a model:

      ```bash
      python -m model_training.train --data-splits data/splits/...
      ```

---

## 9. Code Quality / Style

1. **Run a linter (flake8) with strict rules**

   - Add a `.flake8` (or `setup.cfg`) with at least:

     ```
     [flake8]
     max-line-length = 100
     ignore = E203, E266, E501, W503
     exclude =
       .git,
       __pycache__,
       docs,
       .venv
     ```

   - Ensure each `.py` file in `src/` has no lint errors. This will catch unused imports, undefined names (e.g. `self.text_params`), and missing whitespace around operators.

2. **Add mypy type checking (optional)**

   - Annotate your functions with types (e.g. `def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:`).
   - Add a `mypy.ini` that enforces `--strict` or at least `--ignore-missing-imports`. That helps catch “this method returns None instead of DataFrame” or “this argument might be a list, not a dict.”

3. **Remove bare `except Exception` / suppress warnings globally**

   - In many places you catch a broad `ModuleNotFoundError` and set a boolean. Instead, catch precisely what you need (e.g. `except ImportError:`). Otherwise you might hide a real bug.
   - For logging, avoid using bare `print()` statements in library code; instead use `log.info()`, `log.error()`, etc.

4. **Split large files into smaller modules**

   - `feature_engineering.py` (and `data_preparation.py`) are \~200–300 LOC each. You could break them into:

     ```
     src/Feature_Engineering/
       ├── __init__.py
       ├── transformers.py      # FrequencyEncoder, RareCategory, TextLength, Cyclical, etc.
       ├── selection.py        # VarianceThreshold, MI filter, correlation filter
       └── feature_engineering.py  # orchestrates ColumnTransformer + pipeline
     ```

   - That makes it easier to test each module independently (e.g. `tests/test_transformers.py`, `tests/test_selection.py`, `tests/test_pipeline.py`).

5. **Logging vs. print**
   Everywhere you currently do `print("…")` (e.g. in `split_and_baseline.sanity_checks`), switch to `log.info("…")` or `log.error("…")`. That way you can control verbosity with `logging.basicConfig(level=logging.INFO)`.

---

## 10. Example Data / Synthetic Dataset

1. **Add a tiny “toy” CSV under `data/raw/toy.csv`**

   - Possibly 10 rows, 3 numeric columns, 2 categorical, 1 text column.
   - Show how you expect the entire pipeline—Phase 2 through Phase 5.5—to run on this toy file. When new contributors clone, they can do:

     ```bash
     cp data/raw/toy.csv data/raw/users.csv
     dvc pull
     python -m Data_Ingestion.omni_cli file data/raw/users.csv --save
     python -m ml_pipeline.prepare
     python -m data_analysis.EDA --mode all --target is_churn
     python -m Feature_Engineering.feature_engineering --data data/interim/clean.parquet --target is_churn
     python -m Data_Cleaning.split_and_baseline --target is_churn
     ```

   - This ensures “the repo just works” out of the box on minimal data, rather than waiting for you to configure connections to real S3 buckets or Kafka clusters.

---

## 18. Final Touches

1. **Consistency between `FeatureEngineer` constructor parameter names and README**

   - In your README you call it `numeric_scaler="robust"`, `numeric_power="yeo"`, `log_cols=["revenue"]`, `quantile_bins={"age":4}`, `cat_encoder="hash"`, but in code it’s `cat_encoding`.
   - **Pick one**—either rename the constructor argument to match the README (`cat_encoder`) or update the README to say `cat_encoding`. Same for `text_vectorizer` vs your doc’s mention of `text_vectorizer="tfidf"`.

2. **Raise informative errors when mandatory parameters are missing**

   - If someone calls `python -m Feature_Engineering.feature_engineering --data data/processed/scaled.parquet` but forgets `--target is_churn`, your code will break at `.fit(df, df[None])`. Instead, do:

     ```python
     if args.target is None:
         raise ValueError("`--target` is required. Please specify your target column.")
     ```

### Recap of Major Improvement Areas

- **Code Quality:**

  1. Lint with `flake8`/`black`/`mypy`.
  2. Remove unused parameters (e.g. `interactions`, `self.text_params`).
  3. Add missing pieces (DatetimeExpand, aggregations, interactions).
  4. Fix “oversample” condition, stratified splitting bugs.
  5. Add more granular EDA pipeline (Phase 4.5), share code between EDA scripts.

Perfect. I’ll now rebuild and enhance all seven sections—including 4.5 and 5.5—as modular, class-based components that pass data appropriately (in-memory or serialized where necessary), forming a fully connected pipeline.

I’ll prioritize code quality, robustness, and modular design without CLI scripts or orchestrators. Expect modular Python classes that can be called in sequence and extended easily.

I’ll get started and follow up with a detailed code layout and suggested improvements soon.

# MLDLC Pipeline Implementation

## Section 1: Data Ingestion

```python
import pandas as pd

class DataIngestion:
    """
    Section 1: Data Ingestion.
    Responsible for loading data from source (e.g., CSV file or database) into a pandas DataFrame.
    """
    def __init__(self, source: str, source_type: str = 'csv', **kwargs):
        """
        Initialize DataIngestion.
        :param source: Path to the data file or other source identifier.
        :param source_type: Type of source ('csv' or 'excel' or others if extended).
        :param **kwargs: Additional arguments to pass to pandas read functions (like sep, header, etc).
        """
        self.source = source
        self.source_type = source_type
        self.read_kwargs = kwargs

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the specified source and return a pandas DataFrame.
        Raises FileNotFoundError if the source path is not found.
        """
        if self.source_type == 'csv':
            try:
                df = pd.read_csv(self.source, **self.read_kwargs)
            except FileNotFoundError as e:
                raise e
        elif self.source_type == 'excel':
            try:
                df = pd.read_excel(self.source, **self.read_kwargs)
            except FileNotFoundError as e:
                raise e
        else:
            # Extend for other source types as needed (SQL, etc.)
            raise ValueError(f"Unsupported source_type: {self.source_type}")

        # Basic verification
        if df is None or df.empty:
            raise ValueError("Loaded data is empty.")
        return df
```

## Section 2: Data Cleaning

```python
import pandas as pd
import numpy as np

class DataCleaning:
    """
    Section 2: Data Cleaning.
    Handles missing values, duplicates, and basic data sanity checks/cleaning.
    """
    def __init__(self,
                 missing_values_strategy: str = 'mean',
                 fill_value: float = None,
                 drop_threshold: float = None):
        """
        Initialize DataCleaning.
        :param missing_values_strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop', or 'constant').
        :param fill_value: If strategy is 'constant', use this value to fill missing.
        :param drop_threshold: If provided, drop any columns with missing fraction > drop_threshold (0-1 range).
        """
        self.missing_values_strategy = missing_values_strategy
        self.fill_value = fill_value
        self.drop_threshold = drop_threshold

    def clean(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Clean the dataframe by handling missing values and duplicates.
        :param df: Input DataFrame.
        :param target_col: Optional name of target column. If specified, treat it separately (e.g., drop rows with missing target).
        :return: Cleaned DataFrame.
        """
        df_clean = df.copy()
        # Remove duplicate rows
        df_clean.drop_duplicates(inplace=True)

        # Drop rows where target is missing (if target_col specified)
        if target_col and df_clean[target_col].isna().any():
            df_clean = df_clean[~df_clean[target_col].isna()]

        # Drop columns with too many missing values
        if self.drop_threshold is not None:
            missing_frac = df_clean.isna().mean()
            cols_to_drop = missing_frac[missing_frac > self.drop_threshold].index
            df_clean.drop(columns=cols_to_drop, inplace=True)

        # Fill or drop missing values for each column
        for col in df_clean.columns:
            if col == target_col:
                continue  # skip target for imputation
            if df_clean[col].isna().any():
                if self.missing_values_strategy == 'drop':
                    # drop any rows with missing in this column
                    df_clean = df_clean[~df_clean[col].isna()]
                    continue
                if df_clean[col].dtype == object or str(df_clean[col].dtype).startswith('category'):
                    # Categorical column missing handling
                    if self.missing_values_strategy in ['mode', 'mean', 'median']:
                        # use mode for categorical
                        mode_val = df_clean[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if not mode_val.empty else None
                    elif self.missing_values_strategy == 'constant':
                        fill_val = self.fill_value
                    else:
                        # default for any other unspecified strategy: use mode
                        mode_val = df_clean[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if not mode_val.empty else None
                    df_clean[col].fillna(fill_val, inplace=True)
                else:
                    # Numeric column missing handling
                    if self.missing_values_strategy == 'mean':
                        fill_val = df_clean[col].mean()
                    elif self.missing_values_strategy == 'median':
                        fill_val = df_clean[col].median()
                    elif self.missing_values_strategy == 'mode':
                        mode_val = df_clean[col].mode(dropna=True)
                        fill_val = mode_val.iloc[0] if not mode_val.empty else df_clean[col].mean()
                    elif self.missing_values_strategy == 'constant':
                        fill_val = self.fill_value
                    else:
                        raise ValueError(f"Unknown missing_values_strategy: {self.missing_values_strategy}")
                    df_clean[col].fillna(fill_val, inplace=True)

        # Reset index after dropping rows
        df_clean.reset_index(drop=True, inplace=True)
        return df_clean
```

## Section 3: Exploratory Data Analysis

```python
import pandas as pd
import numpy as np

class ExploratoryDataAnalysis:
    """
    Section 3: Exploratory Data Analysis (EDA).
    Provides methods to summarize data and detect potential issues or insights.
    """
    def __init__(self):
        """Initialize EDA stage."""
        self.report = {}

    def analyze(self, df: pd.DataFrame, target_col: str = None):
        """
        Analyze the dataset and populate a report with summary statistics.
        If target_col is provided, include target distribution and correlations.
        :param df: Input DataFrame.
        :param target_col: Optional target column name for analysis.
        :return: The input DataFrame (unchanged), for pipeline chaining.
        """
        # Basic dataset shape and types
        num_rows, num_cols = df.shape
        dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        self.report['num_rows'] = num_rows
        self.report['num_cols'] = num_cols
        self.report['column_types'] = dtypes

        # Missing value summary
        missing_counts = df.isna().sum().to_dict()
        self.report['missing_values'] = missing_counts

        # Target column analysis if provided
        if target_col:
            if target_col not in df.columns:
                raise ValueError(f"target_col '{target_col}' not in DataFrame columns")
            target = df[target_col]
            # Distribution of target
            if target.nunique() <= 20:
                # likely categorical or discrete
                target_counts = target.value_counts(dropna=False).to_dict()
                self.report['target_distribution'] = target_counts
            else:
                # likely continuous
                self.report['target_mean'] = target.mean()
                self.report['target_std'] = target.std()
                self.report['target_min'] = target.min()
                self.report['target_max'] = target.max()

        # Basic statistics for numeric features
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_dict()
            self.report['numeric_summary'] = stats

        # Categorical feature summary (top categories)
        categorical_cols = [col for col in df.columns if df[col].dtype == object or str(df[col].dtype).startswith('category')]
        cat_summary = {}
        for col in categorical_cols:
            top_vals = df[col].value_counts(dropna=False).head(5).to_dict()
            cat_summary[col] = top_vals
        if cat_summary:
            self.report['categorical_summary'] = cat_summary

        # Correlation with target (for numeric features)
        if target_col and target_col in numeric_cols:
            # if target is numeric, compute Pearson correlation for numeric features
            correlations = {}
            for col in numeric_cols:
                if col == target_col:
                    continue
                correlations[col] = df[col].corr(df[target_col])
            self.report['correlation_with_target'] = correlations
        elif target_col:
            # if target is categorical (classification), compute a simple variance ratio for numeric features
            correlations = {}
            target = df[target_col]
            if target.nunique() > 1:
                for col in numeric_cols:
                    # use an ANOVA-like variance ratio as correlation measure
                    if df[col].nunique() > 0:
                        overall_var = np.var(df[col].dropna().values)
                        within_var = 0
                        n_total = 0
                        for val in target.unique():
                            grp = df[target == val][col].values
                            n = len(grp)
                            n_total += n
                            if n > 1:
                                within_var += n * np.var(grp)
                        if n_total > 0:
                            within_var = within_var / n_total
                            corr_ratio = 1 - within_var/overall_var if overall_var != 0 else 0
                        else:
                            corr_ratio = 0
                        correlations[col] = corr_ratio
                self.report['numeric_feature_correlation_ratio_with_target'] = correlations
            # (Categorical vs categorical correlation not implemented for brevity)

        # Return the original DataFrame to allow chaining
        return df
```

## Section 4: Feature Engineering

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

class FeatureEngineering:
    """
    Section 4: Feature Engineering.
    Encodes categorical features, scales numerical features, and can perform feature selection.
    """
    def __init__(self,
                 categorical_encoding: str = 'onehot',
                 scaling: str = 'standard',
                 drop_constant: bool = True,
                 correlation_threshold: float = None):
        """
        Initialize FeatureEngineering.
        :param categorical_encoding: Encoding method for categorical features ('onehot', 'ordinal', or None for no encoding).
        :param scaling: Scaling method for numeric features ('standard' for StandardScaler, 'minmax' for MinMaxScaler, or None).
        :param drop_constant: If True, drop features with zero variance (constants).
        :param correlation_threshold: If set (0-1), drop one feature in any pair with correlation higher than this threshold.
        """
        self.categorical_encoding = categorical_encoding
        self.scaling = scaling
        self.drop_constant = drop_constant
        self.correlation_threshold = correlation_threshold
        # Attributes to be set in fit
        self.numeric_cols = []
        self.categorical_cols = []
        self.encoder = None
        self.scaler = None
        self.dropped_columns = []
        self.ordinal_categories = {}
        self.target_name = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit the encoders/scalers on X and transform X.
        :param X: Training feature DataFrame.
        :param y: (Optional) target Series for tasks where needed (not used for current encodings).
        :return: Transformed DataFrame.
        """
        X_ft = X.copy()
        # Remove target column from features if present
        if y is not None:
            self.target_name = y.name
            if self.target_name in X_ft.columns:
                X_ft.drop(columns=[self.target_name], inplace=True)
        # Identify numeric and categorical columns
        self.numeric_cols = list(X_ft.select_dtypes(include=np.number).columns)
        self.categorical_cols = [col for col in X_ft.columns if X_ft[col].dtype == object or str(X_ft[col].dtype).startswith('category')]

        # Drop constant features
        if self.drop_constant:
            for col in list(X_ft.columns):
                if X_ft[col].nunique(dropna=False) <= 1:
                    X_ft.drop(columns=[col], inplace=True)
                    self.dropped_columns.append(col)
                    if col in self.numeric_cols:
                        self.numeric_cols.remove(col)
                    if col in self.categorical_cols:
                        self.categorical_cols.remove(col)

        # Drop highly correlated features (for numeric features only)
        if self.correlation_threshold and self.numeric_cols:
            corr_matrix = X_ft[self.numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
            for col in to_drop:
                if col in X_ft.columns:
                    X_ft.drop(columns=[col], inplace=True)
                    self.dropped_columns.append(col)
                    if col in self.numeric_cols:
                        self.numeric_cols.remove(col)
                    if col in self.categorical_cols:
                        self.categorical_cols.remove(col)

        # Handle categorical encoding
        if self.categorical_encoding == 'onehot' and self.categorical_cols:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_array = self.encoder.fit_transform(X_ft[self.categorical_cols])
            # Create column names for one-hot encoded features
            encoded_cols = []
            for i, col in enumerate(self.categorical_cols):
                categories = self.encoder.categories_[i]
                for cat in categories:
                    encoded_cols.append(f"{col}_{cat}")
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X_ft.index)
            # Drop original categorical columns and add new ones
            X_ft.drop(columns=self.categorical_cols, inplace=True)
            X_ft = pd.concat([X_ft, encoded_df], axis=1)
        elif self.categorical_encoding == 'ordinal' and self.categorical_cols:
            for col in self.categorical_cols:
                X_ft[col] = X_ft[col].astype('category')
                self.ordinal_categories[col] = list(X_ft[col].cat.categories)
                # replace categories with codes
                X_ft[col] = X_ft[col].cat.codes
        # if categorical_encoding is None or no categorical columns, do nothing further

        # Scaling numeric features
        if self.scaling and self.numeric_cols:
            if self.scaling == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling}")
            # Fit on current numeric columns (some might have been dropped)
            current_numeric_cols = [col for col in self.numeric_cols if col in X_ft.columns]
            X_ft[current_numeric_cols] = self.scaler.fit_transform(X_ft[current_numeric_cols])

        return X_ft

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a new dataset using the transformations fit on the training data.
        :param X: DataFrame of features to transform (e.g., validation or test set).
        :return: Transformed DataFrame.
        """
        X_t = X.copy()
        # Remove target column if present
        if self.target_name and self.target_name in X_t.columns:
            X_t.drop(columns=[self.target_name], inplace=True)
        # Drop any columns that were dropped during fit
        for col in self.dropped_columns:
            if col in X_t.columns:
                X_t.drop(columns=[col], inplace=True)
        # One-hot encode or ordinal encode as per fit
        if self.categorical_encoding == 'onehot' and self.encoder:
            encoded_array = self.encoder.transform(X_t[self.encoder.feature_names_in_])
            # Create one-hot columns
            encoded_cols = []
            for i, col in enumerate(self.encoder.feature_names_in_):
                for cat in self.encoder.categories_[i]:
                    encoded_cols.append(f"{col}_{cat}")
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=X_t.index)
            X_t.drop(columns=self.encoder.feature_names_in_, inplace=True)
            X_t = pd.concat([X_t, encoded_df], axis=1)
        elif self.categorical_encoding == 'ordinal' and self.ordinal_categories:
            for col, categories in self.ordinal_categories.items():
                if col in X_t.columns:
                    X_t[col] = X_t[col].astype('category')
                    X_t[col] = X_t[col].cat.set_categories(categories)
                    X_t[col] = X_t[col].cat.codes
        # Scale numeric features
        if self.scaler:
            current_numeric_cols = [col for col in self.numeric_cols if col in X_t.columns]
            X_t[current_numeric_cols] = self.scaler.transform(X_t[current_numeric_cols])
        return X_t
```

## Section 4.5: Leakage Detection

```python
import pandas as pd
import numpy as np

class LeakageDetection:
    """
    Section 4.5: Leakage Detection.
    Checks for potential data leakage issues such as target leakage in features or improper data splits.
    """
    def __init__(self):
        self.leakage_report = {}

    def check(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame = None, y_test: pd.Series = None):
        """
        Perform checks for data leakage.
        Returns a dict with findings.
        """
        report = {}
        # Target leakage check: features too correlated with target
        if not y_train.empty:
            if (y_train.dtype in [float, int]) and (y_train.nunique() > 2):
                # Regression target
                corr_with_target = {}
                for col in X_train.select_dtypes(include=np.number).columns:
                    if X_train[col].var() == 0:
                        continue
                    corr = np.corrcoef(X_train[col], y_train)[0,1]
                    if abs(corr) > 0.99:
                        corr_with_target[col] = corr
                if corr_with_target:
                    report['high_corr_features_target'] = corr_with_target
            else:
                # Classification target
                potential_leaks = {}
                for col in X_train.columns:
                    data = pd.DataFrame({col: X_train[col], 'target': y_train})
                    if data[col].dtype.kind in 'if' and data[col].nunique() > 10:
                        data['bin'] = pd.qcut(data[col], q=min(10, len(data)), duplicates='drop')
                        groups = data.groupby('bin')['target']
                    else:
                        groups = data.groupby(col)['target']
                    for val, grp in groups:
                        if len(grp) > 0 and grp.nunique(dropna=False) == 1:
                            potential_leaks.setdefault(col, []).append(val)
                if potential_leaks:
                    report['potential_leakage_features'] = potential_leaks

        # Overlap leakage: check if any identical feature rows exist in both train and test
        if X_test is not None:
            common = pd.merge(X_train, X_test, how='inner')
            if not common.empty:
                report['overlap_rows_train_test'] = len(common)

        # Unseen category check: if test has categories not present in train (not a leakage, but important for model)
        if X_test is not None:
            cat_cols = [col for col in X_train.columns if X_train[col].dtype == object or str(X_train[col].dtype).startswith('category')]
            unseen = {}
            for col in cat_cols:
                if col in X_test.columns:
                    train_vals = set(X_train[col].dropna().unique())
                    test_vals = set(X_test[col].dropna().unique())
                    diff = test_vals - train_vals
                    if diff:
                        unseen[col] = diff
            if unseen:
                report['unseen_test_categories'] = unseen

        self.leakage_report = report
        return report
```

## Section 5: Data Splitting

```python
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    """
    Section 5: Data Splitting.
    Splits the dataset into training, validation (optional), and testing sets, ensuring no data leakage.
    """
    def __init__(self, test_size: float = 0.2, val_size: float = 0.0, stratify: bool = True, random_state: int = None):
        """
        Initialize DataSplitter.
        :param test_size: Proportion of data to allocate to the test set.
        :param val_size: Proportion of data to allocate to the validation set (0 means no validation set).
        :param stratify: Whether to stratify splits by the target class (for classification tasks).
        :param random_state: Random seed for reproducibility.
        """
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.random_state = random_state

    def split(self, df: pd.DataFrame, target_col: str):
        """
        Split the DataFrame into train, validation, and test sets.
        :param df: Input DataFrame containing features and target.
        :param target_col: Name of the target column.
        :return: (X_train, X_val, X_test, y_train, y_val, y_test) if val_size > 0, otherwise (X_train, None, X_test, y_train, None, y_test).

        Example:
            >>> splitter = DataSplitter(test_size=0.2, val_size=0.1, stratify=True, random_state=42)
            >>> X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(df, target_col='target')
            >>> assert len(X_train) + len(X_val) + len(X_test) == len(df)
            >>> assert set(X_train.index).isdisjoint(X_test.index)
        """
        if target_col not in df.columns:
            raise ValueError("target_col not in DataFrame")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        strat = y if self.stratify and (y.nunique() < len(y)) else None
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=self.test_size, stratify=strat, random_state=self.random_state)
        if self.val_size and self.val_size > 0:
            val_fraction = self.val_size / (1 - self.test_size)
            strat2 = y_temp if self.stratify and (y_temp.nunique() < len(y_temp)) else None
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_fraction, stratify=strat2, random_state=self.random_state)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_temp, None, X_test, y_temp, None, y_test
```

## Section 5.5: Baseline Model

```python
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error

class BaselineModel:
    """
    Section 5.5: Baseline Model.
    Trains a simple baseline model (e.g., predicting the mean for regression or most frequent class for classification) and evaluates it.
    """
    def __init__(self, strategy: str = 'most_frequent'):
        """
        Initialize BaselineModel.
        :param strategy: Strategy for baseline (for classification, 'most_frequent' or 'stratified'; for regression, 'mean' or 'median').
        """
        self.strategy = strategy
        self.model = None
        self.is_classification = None
        self.metrics = {}

    def train_and_evaluate(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the baseline model on the training set and evaluate on validation (or test if no val provided).
        :param X_train: Training features.
        :param y_train: Training target.
        :param X_val: Validation features (optional).
        :param y_val: Validation target (optional).
        :return: Trained baseline model and a dict of evaluation metrics.
        """
        # Determine if classification or regression based on y_train
        if y_train.dtype == object or str(y_train.dtype).startswith('category') or y_train.nunique() < 20:
            # If dtype is object/categorical or few unique values, assume classification
            self.is_classification = True
        else:
            self.is_classification = False
        # Initialize dummy model
        if self.is_classification:
            strategy = self.strategy if self.strategy in ['most_frequent','stratified','prior'] else 'most_frequent'
            self.model = DummyClassifier(strategy=strategy)
        else:
            strategy = self.strategy if self.strategy in ['mean','median'] else 'mean'
            self.model = DummyRegressor(strategy=strategy)
        # Train the model
        self.model.fit(X_train, y_train)
        # Evaluate on validation or hold-out set
        X_eval = X_val if X_val is not None else X_train
        y_eval = y_val if y_val is not None else y_train
        y_pred = self.model.predict(X_eval)
        # Compute metrics
        if self.is_classification:
            acc = accuracy_score(y_eval, y_pred)
            average = 'binary' if len(set(y_eval)) == 2 else 'macro'
            f1 = f1_score(y_eval, y_pred, average=average)
            self.metrics = {'accuracy': acc, 'f1': f1}
        else:
            mse = mean_squared_error(y_eval, y_pred)
            mae = mean_absolute_error(y_eval, y_pred)
            self.metrics = {'mse': mse, 'mae': mae}
        return self.model, self.metrics
```

## Section 6: Model Training

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

class ModelTraining:
    """
    Section 6: Model Training.
    Trains a machine learning model (with optional hyperparameter tuning) on the training data.
    """
    def __init__(self, model=None, model_type: str = 'auto', param_grid: dict = None, cv: int = 5, scoring: str = None, random_state: int = None):
        """
        Initialize ModelTraining.
        :param model: An sklearn model instance to train (optional). If provided, model_type is ignored.
        :param model_type: If no model provided, type of model to use ('auto', 'random_forest', 'logistic').
        :param param_grid: Dict of hyperparameters for GridSearchCV (optional).
        :param cv: Number of cross-validation folds for GridSearchCV (if param_grid is provided).
        :param scoring: Scoring metric for GridSearchCV (e.g., 'accuracy', 'f1', 'neg_mean_squared_error', etc.). If None, defaults to estimator's default.
        :param random_state: Random seed (used for reproducible model initialization if applicable).
        """
        self.model = model
        self.model_type = model_type
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.is_classification = None
        self.best_model = None
        self.metrics = {}

    def train_and_evaluate(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model (with hyperparameter tuning if specified) and evaluate on validation set if provided.
        :param X_train: Training features.
        :param y_train: Training target.
        :param X_val: Validation features (optional).
        :param y_val: Validation target (optional).
        :return: Trained model (possibly the best estimator from GridSearchCV) and a dict of performance metrics on validation (or training if no val).
        """
        # Determine classification or regression by target
        if y_train.dtype == object or str(y_train.dtype).startswith('category') or (y_train.dtype != object and y_train.nunique() < 20):
            self.is_classification = True
        else:
            self.is_classification = False

        # Select or create model if not provided
        model = self.model
        if model is None:
            if self.is_classification:
                if self.model_type in ['auto', 'random_forest']:
                    model = RandomForestClassifier(random_state=self.random_state)
                elif self.model_type == 'logistic':
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = RandomForestClassifier(random_state=self.random_state)
            else:
                if self.model_type in ['auto', 'random_forest']:
                    model = RandomForestRegressor(random_state=self.random_state)
                elif self.model_type == 'logistic':
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(random_state=self.random_state)

        # Hyperparameter tuning if param_grid is provided
        if self.param_grid:
            grid = GridSearchCV(model, self.param_grid, cv=self.cv, scoring=self.scoring, n_jobs=-1, refit=True)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)

        self.best_model = best_model

        # Evaluate on validation set if provided, otherwise on training set
        X_eval = X_val if X_val is not None else X_train
        y_eval = y_val if y_val is not None else y_train
        y_pred = best_model.predict(X_eval)
        if self.is_classification:
            acc = accuracy_score(y_eval, y_pred)
            average = 'binary' if len(set(y_eval)) == 2 else 'macro'
            f1 = f1_score(y_eval, y_pred, average=average)
            self.metrics = {'accuracy': acc, 'f1': f1}
        else:
            mse = mean_squared_error(y_eval, y_pred)
            mae = mean_absolute_error(y_eval, y_pred)
            r2 = r2_score(y_eval, y_pred)
            self.metrics = {'mse': mse, 'mae': mae, 'r2': r2}
        return self.best_model, self.metrics
```

## Section 7: Model Evaluation

```python
import joblib
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluation:
    """
    Section 7: Model Evaluation and Deployment.
    Evaluates the trained model on the test set and handles model saving for deployment.
    """
    def __init__(self):
        """Initialize ModelEvaluation."""
        self.results = {}

    def evaluate(self, model, X_test, y_test, baseline_model=None):
        """
        Evaluate the model on the test set, optionally comparing with a baseline model.
        :param model: Trained model to evaluate.
        :param X_test: Test features.
        :param y_test: Test target.
        :param baseline_model: A baseline model for comparison (optional).
        :return: Dictionary of evaluation metrics (for model and baseline if provided).
        """
        results = {}
        # Evaluate main model
        y_pred = model.predict(X_test)
        if y_test.dtype == object or str(y_test.dtype).startswith('category') or (y_test.dtype != object and y_test.nunique() < 20):
            # Classification metrics
            acc = accuracy_score(y_test, y_pred)
            average = 'binary' if len(set(y_test)) == 2 else 'macro'
            f1 = f1_score(y_test, y_pred, average=average)
            results['model_accuracy'] = acc
            results['model_f1'] = f1
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results['model_mse'] = mse
            results['model_mae'] = mae
            results['model_r2'] = r2
        # Evaluate baseline model if given
        if baseline_model is not None:
            y_base_pred = baseline_model.predict(X_test)
            if 'model_accuracy' in results:
                # classification
                base_acc = accuracy_score(y_test, y_base_pred)
                base_f1 = f1_score(y_test, y_base_pred, average=( 'binary' if len(set(y_test)) == 2 else 'macro'))
                results['baseline_accuracy'] = base_acc
                results['baseline_f1'] = base_f1
                results['accuracy_improvement'] = results['model_accuracy'] - base_acc
                results['f1_improvement'] = results['model_f1'] - base_f1
            else:
                # regression
                base_mse = mean_squared_error(y_test, y_base_pred)
                base_mae = mean_absolute_error(y_test, y_base_pred)
                base_r2 = r2_score(y_test, y_base_pred)
                results['baseline_mse'] = base_mse
                results['baseline_mae'] = base_mae
                results['baseline_r2'] = base_r2
                results['mse_reduction'] = base_mse - results['model_mse']
                results['mae_reduction'] = base_mae - results['model_mae']
                results['r2_improvement'] = results['model_r2'] - base_r2
        self.results = results
        return results

    def save_model(self, model, file_path: str):
        """
        Save the trained model to disk for deployment.
        :param model: The model object to save (should be pickle-able).
        :param file_path: File path to save the model (e.g., 'model.pkl').
        """
        joblib.dump(model, file_path)

    @staticmethod
    def load_model(file_path: str):
        """
        Load a model from disk.
        :param file_path: Path to the model file.
        :return: Loaded model object.
        """
        return joblib.load(file_path)
```
