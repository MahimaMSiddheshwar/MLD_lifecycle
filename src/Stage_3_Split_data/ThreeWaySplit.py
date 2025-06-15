#!/usr/bin/env python3
"""
split_and_baseline.py

Phase 5½: Split the (already‐processed) Parquet, optionally SMOTE, train multiple baseline models,
pick the best, freeze the winning baseline (and store checksums), run sanity checks, and freeze the preprocessor.
"""

from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from deepchecks.tabular import TrainTestValidation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# For baseline models
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score

log = None  # you can configure Python logging if desired


class SplitThreeWay:
    def __init__(
        self,
        target: str,
        seed: int = 42,
        stratify: bool = True,
        oversample: bool = False,
    ):
        self.target = target
        self.seed = seed
        self.stratify = stratify
        self.oversample = oversample

        # ─── Paths ─────────────────────────────────────────────────────────────
        self.PROC = Path("data/processed/scaled.parquet")
        self.SPLIT_DIR = Path("data/splits")
        self.SPLIT_DIR.mkdir(parents=True, exist_ok=True)

        self.REPORT_DIR = Path("reports/baseline")
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)

        self.MODEL_DIR = Path("models")
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        self.BASELINE_DIR = self.MODEL_DIR / "baselines"
        self.BASELINE_DIR.mkdir(parents=True, exist_ok=True)

        # ─── Load processed data ─────────────────────────────────────────────────
        if not self.PROC.exists():
            raise FileNotFoundError(f"Expected processed data at {self.PROC}")
        self.df = pd.read_parquet(self.PROC)

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        1) Stratified Train/Val/Test split (80/10/10 by default).
        2) If oversample=True and target is classification, apply SMOTE on the training fold only.
        3) Persist train/val/test in Parquet and write a JSON manifest.
        4) Run Deepchecks TrainTestValidation suite.
        """
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import TrainTestValidation

        y = self.df[self.target]
        stratify_key = y if self.stratify else None

        # 80/20 split
        train, temp = train_test_split(
            self.df,
            test_size=0.20,
            random_state=self.seed,
            stratify=stratify_key,
        )

        stratify_temp = y.loc[temp.index] if self.stratify else None
        val, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=self.seed,
            stratify=stratify_temp,
        )

        # SMOTE oversampling (only for classification)
        if self.oversample and not pd.api.types.is_float_dtype(y.dtype):
            X_tr = train.drop(columns=self.target)
            y_tr = train[self.target]
            X_tr_res, y_tr_res = SMOTE(
                random_state=self.seed).fit_resample(X_tr, y_tr)
            train = pd.concat([X_tr_res, y_tr_res], axis=1)

        # Save splits
        train.to_parquet(self.SPLIT_DIR / "train.parquet", index=False)
        val.to_parquet(self.SPLIT_DIR / "val.parquet", index=False)
        test.to_parquet(self.SPLIT_DIR / "test.parquet", index=False)

        # Save manifest
        manifest = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "seed": self.seed,
            "stratify": self.stratify,
            "oversample": self.oversample,
            "target": self.target,
            "rows": {"train": len(train), "val": len(val), "test": len(test)},
        }
        with open(self.SPLIT_DIR / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # 🧪 Run Deepchecks validation suite on Train vs Test
        cat_features = train.select_dtypes(include="object").columns.tolist()
        train_ds = Dataset(train, label=self.target, cat_features=cat_features)
        test_ds = Dataset(test, label=self.target, cat_features=cat_features)

        suite = TrainTestValidation()
        result = suite.run(train_ds, test_ds)

        result_html = self.SPLIT_DIR / "deepchecks_train_test_validation.html"
        result.save_as_html(str(result_html))

        # Optional: log to MLflow if available
        try:
            import mlflow
            mlflow.log_artifact(str(result_html))
        except Exception:
            pass

        return train, val, test

    def sanity_checks(self) -> None:
        """
        5·3: Basic pipeline sanity checks:
          - No duplicate rows between train/test (on index)
          - No column identical to target (simple leakage sniff)
        """
        tr = pd.read_parquet(self.SPLIT_DIR / "train.parquet")
        te = pd.read_parquet(self.SPLIT_DIR / "test.parquet")

        # Duplicate index check
        dup = set(tr.index).intersection(te.index)
        if dup:
            raise AssertionError(f"Duplicate rows across splits: {len(dup)}")

        # Identical feature → Target (leakage)
        leaks = [
            c for c in tr.columns
            if c != self.target and tr[c].equals(tr[self.target])
        ]
        if leaks:
            raise AssertionError(f"Potential leakage features: {leaks}")

    def run(self) -> None:
        tr, val, te = self.split_data()
        self.sanity_checks()
        return tr, val, te


"""
def cli():
    parser = argparse.ArgumentParser(prog="SplitThreeWay")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stratify", action="store_true",
                        help="Stratify splits by target")
    parser.add_argument("--oversample", action="store_true",
                        help="SMOTE on train only")
    args = parser.parse_args()

    SplitThreeWay(
        target=args.target,
        seed=args.seed,
        stratify=args.stratify,
        oversample=args.oversample,
    ).run()


if __name__ == "__main__":
    cli()


import json, hashlib, joblib
from pathlib import Path

# 1) Load manifest
BASELINE_DIR = Path("models/baselines")
with open(BASELINE_DIR / "baseline_manifest.json", "r") as f:
    manifest = json.load(f)

winner_path = Path(manifest["model_path"])
expected_sha = manifest["sha256"]

# 2) Recompute SHA to ensure integrity
actual_sha = hashlib.sha256(winner_path.read_bytes()).hexdigest()[:12]
if actual_sha != expected_sha:
    raise RuntimeError(f"Baseline checksum mismatch! Expected {expected_sha}, got {actual_sha}")

# 3) Load the model
best_baseline = joblib.load(winner_path)
# Now you can call best_baseline.predict(X_new)

"""
