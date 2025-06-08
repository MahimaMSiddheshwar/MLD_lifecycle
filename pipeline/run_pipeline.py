#!/usr/bin/env python3
"""
run_pipeline.py

A “one‑stop” driver that either runs **dry‑run** (data diagnostics + all EDA/analysis steps)
or runs the end‑to‑end **full pipeline** (data ingestion → prep → feature selection →
feature engineering → split & baseline → training → evaluation → packaging → deploy).

This script does NOT use params.yaml. Instead, adjust the hard‑coded defaults below.
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURE YOUR DEFAULTS HERE
# ─────────────────────────────────────────────────────────────────────────────

# — Data paths —
RAW_DATA_DIR = "data/raw"             # where raw data will land
INTERIM_DATA_PATH = "data/interim/clean.parquet"
PROCESSED_DATA_PATH = "data/processed/scaled.parquet"
SELECTED_DATA_PATH = "data/processed/selected.parquet"
SPLIT_DIR = "data/splits"
MODEL_DIR = "models"
REPORT_DIR_EDA = "reports/eda"
REPORT_DIR_FEATURE = "reports/feature"
REPORT_DIR_BASELINE = "reports/baseline"
REPORT_DIR_METRICS = "reports/metrics"
DEPLOY_SCRIPT = "deploy/push_to_registry.sh"

# — Common parameters —
TARGET_COLUMN = "is_churn"
SEED = 42
STRATIFY = True    # True/False
OVERSAMPLE = False   # True/False

# — Data‑Ingestion defaults (omni_cli) —
#   If you only have a local CSV, set it here.  Otherwise replace with "sql …", "rest …", etc.
OMNI_CLI_DEFAULT_ARGS = "file data/raw/users.csv --redact-pii --save"

# — Data‑Preparation defaults (ml_pipeline.prepare) —
PREP_DEFAULT_ARGS = {
    "knn":    False,       # True → IterativeImputer; False → median/mode
    "outlier": "iqr",      # choices: iqr, zscore, iso
    "scaler":  "standard",  # choices: standard, robust, yeo
    "balance": "none",     # choices: none, smote, nearmiss
    "target":  TARGET_COLUMN
}

# — Feature‑Selection defaults (feature_select.py) —
FS_DEFAULT_ARGS = {
    "nzv_threshold":  1e-5,
    "corr_threshold": 0.95,
    "mi_quantile":    0.10
}

# — Feature‑Engineering defaults (feature_engineering.py) —
FE_DEFAULT_ARGS = {
    "numeric_scaler": "robust",
    "numeric_power":  "yeo",          # yeo, boxcox, quantile, or None
    "log_cols":       ["revenue"],    # list of numeric columns to log1p
    "quantile_bins":  {"age": 4},      # dict col→num_bins
    "polynomial_degree": 2,
    "interactions":    False,
    "rare_threshold":  0.01,
    "cat_encoding":    "target",      # onehot, ordinal, target, woe, hash, freq, none
    "text_vectorizer": "tfidf",       # tfidf, count, hashing, or None
    "text_cols":      ["review"],     # list of text columns
    "datetime_cols":  ["last_login"],  # list of datetime columns to expand
    "cyclical_cols":  {"hour": 24},   # dict col→period
    "date_delta_cols": {"signup_date": "2023-01-01"},
    "aggregations":   {"customer_id": ["amount_mean", "amount_sum"]},
    "drop_nzv":        True,
    "corr_threshold":  0.95,
    "mi_quantile":     0.10,
    "target":          TARGET_COLUMN
}

# — Split & Baseline defaults (split_and_baseline.py) —
SB_DEFAULT_ARGS = {
    "target":   TARGET_COLUMN,
    "seed":     SEED,
    "stratify": STRATIFY,
    "oversample": OVERSAMPLE
}

# — Training defaults (model.train) —
TRAIN_DEFAULT_ARGS = ""  # e.g., "--lr 0.01 --n_estimators 100"; or leave blank

# — Evaluation defaults (model.evaluate) —
EVAL_DEFAULT_ARGS = ""   # e.g., "--threshold 0.5"; or leave blank

# — Packaging defaults (model.package) —
PKG_DEFAULT_ARGS = ""    # e.g., "--format onnx"; or leave blank

# — Deployment defaults —
DEPLOY_DEFAULT_ARGS = ""  # e.g., "<registry_url>"; or leave blank

# ─────────────────────────────────────────────────────────────────────────────
# 2) SET UP LOGGER
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("RunPipeline")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# 3) DATA DIAGNOSTICS CLASS (Dry‑Run)
# ─────────────────────────────────────────────────────────────────────────────

class DataDiagnostics:
    """
    Perform quick, automated checks on the interim dataset:
      • Missing‐value summary
      • Class imbalance (for classification target)
      • Numeric skewness & kurtosis
      • IQR‐based outlier rates
    """

    def __init__(self, interim_path: str, target: str):
        self.interim = Path(interim_path)
        self.target = target
        self.df = None

    def load(self) -> bool:
        if not self.interim.exists():
            log.error(f"[Diagnostics] Interim file not found: {self.interim}")
            return False
        try:
            self.df = pd.read_parquet(self.interim)
            log.info(
                f"[Diagnostics] Loaded interim data: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
            return True
        except Exception as exc:
            log.error(f"[Diagnostics] Error loading interim data: {exc}")
            return False

    def missing_value_summary(self):
        null_counts = self.df.isna().sum()
        pct_missing = (null_counts / len(self.df) * 100).round(2)
        summary = pd.DataFrame(
            {"missing_count": null_counts, "missing_pct": pct_missing})
        print("\n--- Missing Value Summary (Top 10) ---")
        print(summary.sort_values("missing_pct", ascending=False).head(10))

    def class_imbalance(self):
        if self.target not in self.df.columns:
            log.warning(
                f"[Diagnostics] Target '{self.target}' not found; skipping class imbalance check.")
            return
        y = self.df[self.target].dropna()
        # If continuous (regression), skip
        if y.dtype.kind in "biufc" and y.nunique() > 10:
            log.info(
                "[Diagnostics] Regression target detected; skipping class imbalance.")
            return
        counts = y.value_counts(normalize=True)
        print("\n--- Class Balance (Normalized) ---")
        print(counts)
        max_ratio = counts.max()
        if max_ratio > 0.6:
            log.warning(
                f"[Diagnostics] High class imbalance: largest class = {max_ratio:.2f}")
        else:
            log.info("[Diagnostics] Class balance within acceptable range.")

    def skewness_kurtosis(self):
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            log.info(
                "[Diagnostics] No numeric columns found; skipping skew/kurtosis.")
            return
        records = []
        for col in num_cols:
            series = self.df[col].dropna()
            skew = float(series.skew())
            kurt = float(series.kurtosis())
            records.append((col, skew, kurt))
        df_stats = pd.DataFrame(
            records, columns=["feature", "skewness", "kurtosis"])
        print("\n--- Numeric Skewness & Kurtosis (Top 10 by |skew|) ---")
        print(
            df_stats.assign(abs_skew=lambda d: d.skewness.abs())
                    .sort_values("abs_skew", ascending=False)
                    .head(10)[["feature", "skewness", "kurtosis"]]
        )

    def outlier_rates(self):
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            log.info("[Diagnostics] No numeric columns; skipping outlier rates.")
            return
        rates = []
        for col in num_cols:
            series = self.df[col].dropna()
            q1, q3 = np.percentile(series, [25, 75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            pct = len(outliers) / len(series) * 100
            rates.append((col, round(pct, 2)))
        df_out = pd.DataFrame(rates, columns=["feature", "pct_outliers"])
        print("\n--- Outlier Rates by IQR (Top 10) ---")
        print(df_out.sort_values("pct_outliers", ascending=False).head(10))

    def run_all(self):
        if not self.load():
            return
        self.missing_value_summary()
        self.class_imbalance()
        self.skewness_kurtosis()
        self.outlier_rates()


# ─────────────────────────────────────────────────────────────────────────────
# 4) RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(cmd: str, desc: str):
    """Run a shell‐subprocess command, log success or failure, continue on errors."""
    print("\n" + "="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶▶ {desc}")
    print(f"    → Command: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        print(f"    ✔ Success\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(
            f"    ✖ Failed (continuing anyway)\n----- stdout/err -----\n{e.stdout}\n----------------------")


def main():
    parser = argparse.ArgumentParser(
        description="“One‑stop” driver: either a dry‑run (data diagnostics + EDA) or full end‑to‑end pipeline."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run data‑quality diagnostics + all EDA/analysis steps, then exit."
    )
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────────────────────
    # If dry‑run: do data diagnostics + EDA + probabilistic + feature selection, then exit.
    # ─────────────────────────────────────────────────────────────────────────
    if args.dry_run:
        log.info("▶ Dry‑run mode: Running diagnostics + EDA/analysis, then exiting.")

        # 1) Data diagnostics on interim dataset
        diag = DataDiagnostics(
            interim_path=INTERIM_DATA_PATH, target=TARGET_COLUMN)
        diag.run_all()

        # 2) Core EDA
        eda_cmd = f"python -m Data_Analysis.EDA --mode all --target {TARGET_COLUMN}"
        run_stage(eda_cmd, "Dry‑run: Core EDA")

        # 3) Advanced EDA
        eda_adv_cmd = "python -m Data_Analysis.EDA_advance"
        run_stage(eda_adv_cmd, "Dry‑run: Advanced EDA")

        # 4) Probabilistic analysis (defaults, no flags)
        prob_cmd = "python -m data_analysis.probabilistic_analysis"
        run_stage(prob_cmd, "Dry‑run: Probabilistic Analysis")

        # 5) Feature selection (defaults; no flags → uses built‑in thresholds)
        fs_cmd = "python -m Feature_Selection.feature_select"
        run_stage(fs_cmd, "Dry‑run: Feature Selection")

        print("\nDry‑run complete. Exiting.\n")
        sys.exit(0)

    # ─────────────────────────────────────────────────────────────────────────
    # Full end‑to‑end pipeline (no params.yaml)
    # ─────────────────────────────────────────────────────────────────────────

    # Phase 2: Data Ingestion
    omni_cmd = f"python -m data_ingest.omni_cli {OMNI_CLI_DEFAULT_ARGS}"
    run_stage(omni_cmd, "Phase 2: Data Collection (omni_cli)")

    # Phase 3: Data Preparation
    prep_args = []
    if PREP_DEFAULT_ARGS["knn"]:
        prep_args.append("--knn")
    prep_args.append(f"--outlier {PREP_DEFAULT_ARGS['outlier']}")
    prep_args.append(f"--scaler {PREP_DEFAULT_ARGS['scaler']}")
    if PREP_DEFAULT_ARGS["balance"] != "none":
        prep_args.append(f"--balance {PREP_DEFAULT_ARGS['balance']}")
    prep_args.append(f"--target {PREP_DEFAULT_ARGS['target']}")
    prep_cmd = "python -m ml_pipeline.prepare " + " ".join(prep_args)
    run_stage(prep_cmd, "Phase 3: Data Preparation")

    # Phase 4: Core EDA
    eda_cmd = f"python -m Data_Analysis.EDA --mode all --target {TARGET_COLUMN}"
    run_stage(eda_cmd, "Phase 4: EDA (core)")

    # Phase 4D: Advanced EDA
    eda_adv_cmd = "python -m Data_Analysis.EDA_advance"
    run_stage(eda_adv_cmd, "Phase 4D: EDA (advanced)")

    # Phase 4½: Probabilistic Analysis
    prob_cmd = "python -m data_analysis.probabilistic_analysis"
    run_stage(prob_cmd, "Phase 4½: Probabilistic Analysis")

    # Phase 4½: Feature Selection
    # Build flags from FS_DEFAULT_ARGS
    fs_flags = []
    if "nzv_threshold" in FS_DEFAULT_ARGS:
        fs_flags.append(f"--nzv_threshold {FS_DEFAULT_ARGS['nzv_threshold']}")
    if "corr_threshold" in FS_DEFAULT_ARGS:
        fs_flags.append(
            f"--corr_threshold {FS_DEFAULT_ARGS['corr_threshold']}")
    if "mi_quantile" in FS_DEFAULT_ARGS:
        fs_flags.append(f"--mi_quantile {FS_DEFAULT_ARGS['mi_quantile']}")
    fs_cmd = "python -m Feature_Selection.feature_select " + " ".join(fs_flags)
    run_stage(fs_cmd, "Phase 4½: Feature Selection")

    # Phase 5: Feature Engineering
    fe_flags = []
    fe_flags.append(f"--data {SELECTED_DATA_PATH}")
    fe_flags.append(f"--target {FE_DEFAULT_ARGS['target']}")
    fe_flags.append(f"--numeric_scaler {FE_DEFAULT_ARGS['numeric_scaler']}")
    if FE_DEFAULT_ARGS["numeric_power"]:
        fe_flags.append(f"--numeric_power {FE_DEFAULT_ARGS['numeric_power']}")
    if FE_DEFAULT_ARGS["log_cols"]:
        fe_flags.append(f"--log_cols {','.join(FE_DEFAULT_ARGS['log_cols'])}")
    if FE_DEFAULT_ARGS["quantile_bins"]:
        qb_list = ";".join(
            f"{k}:{v}" for k, v in FE_DEFAULT_ARGS["quantile_bins"].items())
        fe_flags.append(f"--quantile_bins {qb_list}")
    if FE_DEFAULT_ARGS["polynomial_degree"] is not None:
        fe_flags.append(
            f"--polynomial_degree {FE_DEFAULT_ARGS['polynomial_degree']}")
    if FE_DEFAULT_ARGS["interactions"]:
        fe_flags.append("--interactions")
    if FE_DEFAULT_ARGS["rare_threshold"] is not None:
        fe_flags.append(
            f"--rare_threshold {FE_DEFAULT_ARGS['rare_threshold']}")
    if FE_DEFAULT_ARGS["cat_encoding"]:
        fe_flags.append(f"--cat_encoding {FE_DEFAULT_ARGS['cat_encoding']}")
    if FE_DEFAULT_ARGS["text_vectorizer"]:
        fe_flags.append(
            f"--text_vectorizer {FE_DEFAULT_ARGS['text_vectorizer']}")
    if FE_DEFAULT_ARGS["text_cols"]:
        fe_flags.append(
            f"--text_cols {','.join(FE_DEFAULT_ARGS['text_cols'])}")
    if FE_DEFAULT_ARGS["datetime_cols"]:
        fe_flags.append(
            f"--datetime_cols {','.join(FE_DEFAULT_ARGS['datetime_cols'])}")
    if FE_DEFAULT_ARGS["cyclical_cols"]:
        cyc_list = ";".join(
            f"{k}:{v}" for k, v in FE_DEFAULT_ARGS["cyclical_cols"].items())
        fe_flags.append(f"--cyclical_cols {cyc_list}")
    if FE_DEFAULT_ARGS["date_delta_cols"]:
        dd_list = ";".join(
            f"{k}:{v}" for k, v in FE_DEFAULT_ARGS["date_delta_cols"].items())
        fe_flags.append(f"--date_delta_cols {dd_list}")
    if FE_DEFAULT_ARGS["aggregations"]:
        agg_list = ";".join(f"{k}:{','.join(v)}" for k,
                            v in FE_DEFAULT_ARGS["aggregations"].items())
        fe_flags.append(f"--aggregations {agg_list}")
    if FE_DEFAULT_ARGS["drop_nzv"]:
        fe_flags.append("--drop_nzv")
    if FE_DEFAULT_ARGS["corr_threshold"] is not None:
        fe_flags.append(
            f"--corr_threshold {FE_DEFAULT_ARGS['corr_threshold']}")
    if FE_DEFAULT_ARGS["mi_quantile"] is not None:
        fe_flags.append(f"--mi_quantile {FE_DEFAULT_ARGS['mi_quantile']}")

    fe_cmd = "python -m Feature_Engineering.feature_engineering " + \
        " ".join(fe_flags)
    run_stage(fe_cmd, "Phase 5: Feature Engineering")

    # Phase 5½: Split & Baseline
    sb_flags = []
    sb_flags.append(f"--target {SB_DEFAULT_ARGS['target']}")
    sb_flags.append(f"--seed {SB_DEFAULT_ARGS['seed']}")
    if SB_DEFAULT_ARGS["stratify"]:
        sb_flags.append("--stratify")
    if SB_DEFAULT_ARGS["oversample"]:
        sb_flags.append("--oversample")
    sb_cmd = "python -m Data_Cleaning.split_and_baseline " + " ".join(sb_flags)
    run_stage(sb_cmd, "Phase 5½: Split & Baseline")

    # Phase 6: Training & Tuning
    train_cmd = f"python -m model.train {TRAIN_DEFAULT_ARGS}"
    run_stage(train_cmd, "Phase 6: Model Training / Tuning")

    # Phase 7: Evaluation
    eval_cmd = f"python -m model.evaluate {EVAL_DEFAULT_ARGS}"
    run_stage(eval_cmd, "Phase 7: Evaluation")

    # Phase 8: Packaging
    pkg_cmd = f"python -m model.package {PKG_DEFAULT_ARGS}"
    run_stage(pkg_cmd, "Phase 8: Packaging")

    # Phase 9: Deployment (optional)
    deploy_cmd = f"bash {DEPLOY_SCRIPT} {DEPLOY_DEFAULT_ARGS}"
    run_stage(deploy_cmd, "Phase 9: Deployment (optional)")

    print("\n" + "="*80)
    print("▶ All stages attempted. Check logs above for any errors.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
