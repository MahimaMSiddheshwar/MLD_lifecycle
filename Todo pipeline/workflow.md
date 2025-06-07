# What This Script Does

1. **Phase 2 — Data Collection**

   - Runs

     ```bash
     python -m data_ingest.omni_cli file data/raw/users.csv --redact-pii --save
     ```

   - You can edit this line if your data source is different (e.g. `"sql …"` or `"rest …"`).

2. **Phase 3 — Data Preparation**

   - Runs

     ```bash
     python -m ml_pipeline.prepare --outlier iqr --scaler standard
     ```

   - Default choices: IQR outlier detection, StandardScaler, median/mode imputation, no balancing.
   - Checks that `data/interim/clean.parquet` and `data/processed/scaled.parquet` are produced.

3. **Phase 4 — EDA (Core)**

   - Runs

     ```bash
     python -m Data_Analysis.EDA --mode all --target is_churn --profile
     ```

   - Generates univariate, bivariate, multivariate stats + plots, and an optional HTML profile.

4. **Phase 4D — EDA (Advanced)**

   - Runs

     ```bash
     python -m Data_Analysis.EDA_advance
     ```

   - Provides deeper analyses (mutual information, leakage sniff, TS decor, etc.).

5. **Phase 4½ — Probabilistic Analysis**

   - Runs

     ```bash
     python -m data_analysis.probabilistic_analysis --impute_method mice --target is_churn
     ```

   - You can add `--do_pit`, `--do_quantile`, or `--do_copula` if desired.

6. **Phase 4½ — Feature Selection**

   - Runs

     ```bash
     python -m Feature_Selection.feature_select --data data/processed/scaled.parquet --target is_churn --nzv_threshold 1e-5 --corr_threshold 0.95 --mi_quantile 0.10
     ```

   - Outputs `data/processed/selected.parquet` and `reports/feature/feature_audit.json`.

7. **Phase 5 — Feature Engineering**

   - Runs

     ```bash
     python -m Feature_Engineering.feature_engineering \
       --data data/processed/selected.parquet \
       --target is_churn \
       --numeric_scaler robust \
       --numeric_power yeo \
       --log_cols revenue \
       --quantile_bins age:4 \
       --polynomial_degree 2 \
       --rare_threshold 0.01 \
       --cat_encoding target \
       --text_vectorizer tfidf \
       --text_cols review \
       --datetime_cols last_login \
       --cyclical_cols hour:24 \
       --date_delta_cols signup_date:today \
       --aggregations customer_id:amount_mean,amount_sum \
       --drop_nzv \
       --corr_threshold 0.95 \
       --mi_quantile 0.10
     ```

   - Outputs `models/preprocessor.joblib`, `models/preprocessor.json` (SHA), plus
     `reports/feature/feature_audit.json` and `reports/feature/feature_shape.txt`.

8. **Phase 5½ — Split & Baseline**

   - Runs

     ```bash
     python -m Data_Cleaning.split_and_baseline --target is_churn --seed 42 --stratify
     ```

   - Creates `data/splits/{train.parquet, val.parquet, test.parquet}`,
     `data/splits/split_manifest.json`, and `reports/baseline/baseline_metrics.json`,
     and snapshots `models/preprocessor_manifest.json`.

9. **Phase 6 — Train + Tune**

   - Runs

     ```bash
     python -m model.train
     ```

   - Consumes splits and preprocessor, produces `models/model.pkl`, `models/model_card.md`, and training metrics in `reports/metrics/`.

10. **Phase 7 — Evaluate**

    - Runs

      ```bash
      python -m model.evaluate
      ```

    - Consumes `data/splits/test.parquet` and `models/model.pkl`;
      produces `reports/metrics/test_metrics.json` and `reports/metrics/roc_curve.csv`.

11. **Phase 8 — Package → ONNX**

    - Runs

      ```bash
      python -m model.package --model models/model.pkl
      ```

    - Exports `artefacts/model.onnx` (if implemented).

12. **Phase 9 — Deploy (Optional)**

    - If `deploy/push_to_registry.sh` exists, runs it. Otherwise, logs “SKIP”.

## How to Use

1. **Dry‑Run (Data Diagnostics + EDA + Probabilistic + Feature Selection)**
   This mode will _only_ run diagnostics and analysis on your existing interim dataset (`data/interim/clean.parquet`) and then exit.

   ```bash
   python run_pipeline.py --dry-run
   ```

   - **Data Diagnostics** (missing values, imbalance, skewness, outliers)
   - **Core EDA** → `python -m Data_Analysis.EDA --mode all --target is_churn`
   - **Advanced EDA** → `python -m Data_Analysis.EDA_advance`
   - **Probabilistic Analysis** → `python -m data_analysis.probabilistic_analysis`
   - **Feature Selection** → `python -m Feature_Selection.feature_select`

   > After these steps, the script prints a message and exits without running the rest of the pipeline.

2. **Full Pipeline (End‑to‑End)**
   This mode executes every phase in sequence:

   1. **Phase 2 – Data Collection**

      ```bash
      python -m data_ingest.omni_cli file data/raw/users.csv --redact-pii --save
      ```

   2. **Phase 3 – Data Preparation**

      ```bash
      python -m ml_pipeline.prepare --outlier iqr --scaler standard --target is_churn
      ```

      (Add `--knn` or `--balance smote` if you modify `PREP_DEFAULT_ARGS` to `True`.)

   3. **Phase 4 – Core EDA**

      ```bash
      python -m Data_Analysis.EDA --mode all --target is_churn
      ```

   4. **Phase 4D – Advanced EDA**

      ```bash
      python -m Data_Analysis.EDA_advance
      ```

   5. **Phase 4½ – Probabilistic Analysis**

      ```bash
      python -m data_analysis.probabilistic_analysis
      ```

   6. **Phase 4½ – Feature Selection**

      ```bash
      python -m Feature_Selection.feature_select --nzv_threshold 1e-5 --corr_threshold 0.95 --mi_quantile 0.1
      ```

   7. **Phase 5 – Feature Engineering**

      ```bash
      python -m Feature_Engineering.feature_engineering \
        --data data/processed/selected.parquet \
        --target is_churn \
        --numeric_scaler robust \
        --numeric_power yeo \
        --log_cols revenue \
        --quantile_bins age:4 \
        --polynomial_degree 2 \
        --rare_threshold 0.01 \
        --cat_encoding target \
        --text_vectorizer tfidf \
        --text_cols review \
        --datetime_cols last_login \
        --cyclical_cols hour:24 \
        --date_delta_cols signup_date:2023-01-01 \
        --aggregations customer_id:amount_mean,amount_sum \
        --drop_nzv \
        --corr_threshold 0.95 \
        --mi_quantile 0.1
      ```

   8. **Phase 5½ – Split & Baseline**

      ```bash
      python -m Data_Cleaning.split_and_baseline --target is_churn --seed 42 --stratify
      ```

   9. **Phase 6 – Model Training & Tuning**

      ```bash
      python -m model.train
      ```

      (Add any flags by editing `TRAIN_DEFAULT_ARGS` at top of this script.)

3. **Phase 7 – Evaluation**

   ```bash
   python -m model.evaluate
   ```

4. **Phase 8 – Packaging**

   ```bash
   python -m model.package
   ```

5. **Phase 9 – Deployment**

   ```bash
   bash deploy/push_to_registry.sh
   ```

> Because **`run_pipeline.py`** no longer references `params.yaml`, you can either edit the hard‑coded defaults at the very top of `run_pipeline.py` (e.g. change `TARGET_COLUMN`, adjust `OMNI_CLI_DEFAULT_ARGS`, switch to SMOTE, etc.) or simply let it run with those values as‑is.

---

### Summary of Key Points

- **No `params.yaml` dependency**—all defaults live in `run_pipeline.py`.
- **Dry‑Run mode** (`--dry-run`)
  • Loads `data/interim/clean.parquet` → runs data diagnostics (missingness, imbalance, skew, outliers).
  • Runs core EDA + advanced EDA via Python modules (no need for manual notebook).
  • Runs probabilistic analysis.
  • Runs feature selection.
  • Then exits.
- **Full Pipeline** (no flags)
  • Invokes each phase in order via shell commands—data ingestion → prep → EDA → prob analysis → feature selection → feature engineering → split & baseline → train → evaluate → package → deploy.
