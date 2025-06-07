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


class SplitAndBaseline:
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
        """
        y = self.df[self.target]
        stratify_key = y if self.stratify else None

        # 80/20
        train, temp = train_test_split(
            self.df,
            test_size=0.20,
            random_state=self.seed,
            stratify=stratify_key,
        )

        stratify_temp = y.loc[temp.index] if self.stratify else None
        # split temp 50/50 → 10/10 total
        val, test = train_test_split(
            temp,
            test_size=0.50,
            random_state=self.seed,
            stratify=stratify_temp,
        )

        # 5·1: Optionally SMOTE on training fold only (classification only)
        if self.oversample and not pd.api.types.is_float_dtype(y.dtype):
            X_tr = train.drop(columns=self.target)
            y_tr = train[self.target]
            X_tr_res, y_tr_res = SMOTE(
                random_state=self.seed).fit_resample(X_tr, y_tr)
            train = pd.concat([X_tr_res, y_tr_res], axis=1)

        # Persist splits
        train.to_parquet(self.SPLIT_DIR / "train.parquet", index=False)
        val.to_parquet(self.SPLIT_DIR / "val.parquet", index=False)
        test.to_parquet(self.SPLIT_DIR / "test.parquet", index=False)

        # Write split manifest
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

        return train, val, test

    def build_baseline(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        """
        5·2: Train and evaluate multiple baseline “dummy” models.  
           - For regression: DummyRegressor(["mean", "median", "quantile"])  
           - For classification: DummyClassifier(["most_frequent", "stratified", "uniform"])  
        1) Fit each candidate on the training set.  
        2) Evaluate on validation (or test) set.  
        3) Pick “best” by a chosen metric (lowest MAE for regression; highest F1 for classification).  
        4) Save ALL candidates under models/baselines/, compute their SHA-256.  
        5) Mark the winner in baseline_manifest.json and persist its path + sha.  
        6) Write out baseline_metrics.json summarizing each candidate’s performance.  
        """
        X_tr = train.drop(columns=self.target)
        y_tr = train[self.target]

        X_val = val.drop(columns=self.target)
        y_val = val[self.target]

        results: dict[str, dict] = {}
        best_name = None
        best_score = None

        is_regression = pd.api.types.is_float_dtype(y_tr.dtype)

        if is_regression:
            # ── Candidate Regressors ────────────────────────────────────────────
            candidates = {
                "mean_regressor": DummyRegressor(strategy="mean"),
                "median_regressor": DummyRegressor(strategy="median"),
                # “quantile” with quantile=0.5 is essentially median; use 0.25 here as example
                "quantile0.25_regressor": DummyRegressor(strategy="quantile", quantile=0.25),
            }

            # Evaluate each on val‐set by MAE (and also track R²)
            for name, model in candidates.items():
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                mae = float(mean_absolute_error(y_val, preds))
                r2 = float(r2_score(y_val, preds))
                results[name] = {"MAE": mae, "R2": r2}

                # Save candidate to disk
                filepath = self.BASELINE_DIR / f"{name}.joblib"
                joblib.dump(model, filepath)

                # Compute checksum
                sha = hashlib.sha256(filepath.read_bytes()).hexdigest()[:12]
                results[name]["model_path"] = str(filepath)
                results[name]["sha256"] = sha

                # Determine winner (lowest MAE)
                if best_score is None or mae < best_score:
                    best_score = mae
                    best_name = name

        else:
            # ── Candidate Classifiers ───────────────────────────────────────────
            candidates = {
                "most_frequent_clf": DummyClassifier(strategy="most_frequent"),
                "stratified_clf": DummyClassifier(strategy="stratified", random_state=self.seed),
                "uniform_clf": DummyClassifier(strategy="uniform", random_state=self.seed),
                # “constant” could be added if you want to force‐predict a particular class
            }

            # Evaluate each on val‐set by F1 (and also track accuracy)
            for name, model in candidates.items():
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                acc = float(accuracy_score(y_val, preds))
                f1 = float(f1_score(y_val, preds, zero_division=0))
                results[name] = {"accuracy": acc, "f1": f1}

                # Save candidate to disk
                filepath = self.BASELINE_DIR / f"{name}.joblib"
                joblib.dump(model, filepath)

                # Compute checksum
                sha = hashlib.sha256(filepath.read_bytes()).hexdigest()[:12]
                results[name]["model_path"] = str(filepath)
                results[name]["sha256"] = sha

                # Determine winner (highest F1 by default)
                if best_score is None or f1 > best_score:
                    best_score = f1
                    best_name = name

        # ── Write out detailed baseline_metrics.json ───────────────────────────
        #   Structure:
        #   {
        #     "candidates": {
        #         "mean_regressor": {"MAE": 2.3, "R2": 0.12, "model_path": "...", "sha256": "..."},
        #         "median_regressor": { ... },
        #          …
        #      },
        #     "best": {
        #         "name": "median_regressor",
        #         "metric": {<either "MAE": x> or {"f1": x}}
        #     }
        #   }
        baseline_report = {"candidates": results,
                           "best": {"name": best_name, "metric": {}}}
        if is_regression:
            baseline_report["best"]["metric"] = {
                "MAE": results[best_name]["MAE"], "R2": results[best_name]["R2"]}
        else:
            baseline_report["best"]["metric"] = {
                "accuracy": results[best_name]["accuracy"], "f1": results[best_name]["f1"]}

        # Save JSON
        with open(self.REPORT_DIR / "baseline_metrics.json", "w") as f:
            json.dump(baseline_report, f, indent=2)

        # ── Now create a small “baseline_manifest.json” that only references the winning model and its checksum ──
        winner_path = Path(results[best_name]["model_path"])
        winner_sha = results[best_name]["sha256"]
        manifest = {
            "winner": best_name,
            "model_path": str(winner_path),
            "sha256": winner_sha,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        }
        with open(self.BASELINE_DIR / "baseline_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return

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

    def freeze_preprocessor(self) -> None:
        """
        5·4: Persist the Phase-3 numeric preprocessor (StandardScaler) as a baseline.
        We assume the Phase 3 preprocessor has a single numeric-scaling component.
        """
        num_cols = self.df.select_dtypes(include="number").columns
        scaler_pipe = Pipeline([("scale", StandardScaler())])
        scaler_pipe.fit(self.df[num_cols])

        # Save it
        preproc_path = self.MODEL_DIR / "preprocessor.joblib"
        joblib.dump(scaler_pipe, preproc_path)

        # Compute checksum
        sha = hashlib.sha256(preproc_path.read_bytes()).hexdigest()[:12]
        manifest = {"path": "models/preprocessor.joblib", "sha256": sha,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds")}
        with open(self.MODEL_DIR / "preprocessor_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def run(self) -> None:
        tr, val, te = self.split_data()
        self.build_baseline(tr, val, te)
        self.sanity_checks()
        self.freeze_preprocessor()
        print("🟢 Phase 5½: Split + Multi‐Baseline + Sanity + Preprocessor Freeze complete!")


def cli():
    parser = argparse.ArgumentParser(prog="split_and_baseline")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stratify", action="store_true",
                        help="Stratify splits by target")
    parser.add_argument("--oversample", action="store_true",
                        help="SMOTE on train only")
    args = parser.parse_args()

    SplitAndBaseline(
        target=args.target,
        seed=args.seed,
        stratify=args.stratify,
        oversample=args.oversample,
    ).run()


if __name__ == "__main__":
    cli()


"""
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
