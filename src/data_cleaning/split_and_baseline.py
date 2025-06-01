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


class SplitAndBaseline:
    """
    Orchestrates Dataset Partition & Baseline Benchmarking (Phase 5½):
    5·0 split_data()
    5·1 stratify_or_group()
    5·2 build_baseline()
    5·3 sanity_checks()
    5·4 freeze_preprocessor()
    """

    def __init__(self,
                 target: str,
                 seed: int = 42,
                 stratify: bool = False,
                 oversample: bool = False):
        self.target = target
        self.seed = seed
        self.stratify = stratify
        self.oversample = oversample

        # Paths
        self.PROC = Path("data/processed/scaled.parquet")
        self.SPLIT_DIR = Path("data/splits")
        self.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORT_DIR = Path("reports/baseline")
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR = Path("models")
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Load full processed dataset
        if not self.PROC.exists():
            raise FileNotFoundError(f"Expected processed data at {self.PROC}")
        self.df = pd.read_parquet(self.PROC)

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        5·0 + 5·1: Create train/val/test splits (80/10/10),
        stratified if requested. Optionally SMOTE on train only.
        """
        y = self.df[self.target]
        stratify_key = y if self.stratify else None

        train, temp = train_test_split(
            self.df, test_size=0.2, random_state=self.seed, stratify=stratify_key
        )
        stratify_temp = y.loc[temp.index] if self.stratify else None
        val, test = train_test_split(
            temp, test_size=0.5, random_state=self.seed, stratify=stratify_temp
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

        # Write manifest
        manifest = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "seed": self.seed,
            "stratify": self.stratify,
            "oversample": self.oversample,
            "target": self.target,
            "rows": {"train": len(train), "val": len(val), "test": len(test)}
        }
        with open(self.SPLIT_DIR / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return train, val, test

    def build_baseline(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        """
        5·2: Compute a trivial baseline:
        - If regression: use mean regressor (MAE, R2)  
        - If classification: use majority class (accuracy, F1)
        """
        y_test = test[self.target]
        metrics = {}

        if pd.api.types.is_float_dtype(y_test.dtype):
            # Regression baseline
            pred = np.full_like(y_test, train[self.target].mean(), dtype=float)
            metrics = {
                "type": "mean_regressor",
                "mae": float(np.mean(np.abs(y_test - pred))),
                "r2": float(r2_score(y_test, pred))
            }
        else:
            # Classification baseline
            majority_class = int(train[self.target].mode()[0])
            pred = np.full_like(y_test, majority_class)
            metrics = {
                "type": "majority_class",
                "majority_class": majority_class,
                "accuracy": float((y_test == pred).mean()),
                "f1": float(
                    f1_score(y_test, pred, zero_division=0)
                )
            }

        with open(self.REPORT_DIR / "baseline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

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

        # Identical feature → Target
        leaks = [
            c for c in tr.columns
            if c != self.target and tr[c].equals(tr[self.target])
        ]
        if leaks:
            raise AssertionError(f"Potential leakage features: {leaks}")

    def freeze_preprocessor(self) -> None:
        """
        5·4: Persist the Phase‑3 numeric preprocessor (StandardScaler) as a baseline.
        We assume the Phase 3 preprocessor has a single numeric‑scaling component.
        """
        num_cols = self.df.select_dtypes(include="number").columns
        scaler_pipe = Pipeline([("scale", StandardScaler())])
        scaler_pipe.fit(self.df[num_cols])

        # Save it
        joblib.dump(scaler_pipe, self.MODEL_DIR / "preprocessor.joblib")

        # Checksum + manifest
        sha = hashlib.sha256(
            Path(self.MODEL_DIR / "preprocessor.joblib").read_bytes()
        ).hexdigest()[:12]
        with open(self.MODEL_DIR / "preprocessor_manifest.json", "w") as f:
            json.dump({"path": "models/preprocessor.joblib",
                      "sha256": sha}, f, indent=2)

    def run(self):
        tr, val, te = self.split_data()
        self.build_baseline(tr, te)
        self.sanity_checks()
        self.freeze_preprocessor()
        print("🟢 Phase 5½: Split + Baseline complete!")


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
        oversample=args.oversample
    ).run()


if __name__ == "__main__":
    cli()
