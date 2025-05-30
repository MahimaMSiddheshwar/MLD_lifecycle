"""
src/Data_Cleaning/split_and_baseline.py
---------------------------------------
A single class that covers **ALL** sub-steps of the
“Dataset Partition & Baseline Benchmarking” phase:

    5·0  split_data()
    5·1  stratify_or_group()
    5·2  build_baseline()
    5·3  sanity_checks()
    5·4  freeze_preprocessor()

Run end-to-end:

    python -m Data_Cleaning.split_and_baseline \
           --target is_churn --seed 42 --stratify
"""

from __future__ import annotations
import argparse
import json
import joblib
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------------


class SplitAndBaseline:
    """Orchestrates split, baseline, sanity, and pipeline freeze."""

    def __init__(self,
                 target: str,
                 seed: int = 42,
                 stratify: bool = False,
                 oversample: bool = False):

        self.target = target
        self.seed = seed
        self.stratify = stratify
        self.oversample = oversample

        # paths
        self.PROC = Path("data/processed/scaled.parquet")
        self.SPLIT = Path("data/splits")
        self.SPLIT.mkdir(parents=True, exist_ok=True)
        self.REPORT = Path("reports/baseline")
        self.REPORT.mkdir(parents=True, exist_ok=True)
        self.MODEL = Path("models")
        self.MODEL.mkdir(exist_ok=True)

        # load dataset
        self.df = pd.read_parquet(self.PROC)

    # ----------------------------- 5·0 + 5·1
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        y = self.df[self.target]
        strat = y if self.stratify else None

        train, temp = train_test_split(
            self.df, test_size=0.3, random_state=self.seed, stratify=strat)
        val, test = train_test_split(
            temp, test_size=0.5, random_state=self.seed,
            stratify=strat.loc[temp.index] if self.stratify else None)

        # optional SMOTE on training fold
        if self.oversample and y.dtype.kind not in "if":
            X_tr, y_tr = train.drop(columns=self.target), train[self.target]
            X_tr, y_tr = SMOTE(random_state=self.seed).fit_resample(X_tr, y_tr)
            train = pd.concat([X_tr, y_tr], axis=1)

        train.to_parquet(self.SPLIT/"train.parquet", index=False)
        val.to_parquet(self.SPLIT/"val.parquet", index=False)
        test.to_parquet(self.SPLIT/"test.parquet", index=False)

        # manifest
        manifest = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "seed": self.seed,
            "stratify": self.stratify,
            "oversample": self.oversample,
            "target": self.target,
            "rows": {"train": len(train), "val": len(val), "test": len(test)}
        }
        (self.SPLIT/"split_manifest.json").write_text(json.dumps(manifest, indent=2))
        return train, val, test

    # ----------------------------- 5·2
    def build_baseline(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        y_test = test[self.target]

        if y_test.dtype.kind in "if":                     # regression baseline
            pred = np.full_like(y_test, train[self.target].mean(), dtype=float)
            metrics = {
                "type": "mean_regressor",
                "mae": mean_absolute_error(y_test, pred),
                "r2": r2_score(y_test, pred)
            }
        else:                                            # classification baseline
            majority = train[self.target].mode()[0]
            pred = np.full_like(y_test, majority)
            metrics = {
                "type": "majority_class",
                "majority_class": int(majority),
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred, zero_division=0)
            }

        (self.REPORT/"baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ----------------------------- 5·3
    def sanity_checks(self) -> None:
        tr = pd.read_parquet(self.SPLIT/"train.parquet")
        te = pd.read_parquet(self.SPLIT/"test.parquet")

        # duplicate index check
        dup = set(tr.index).intersection(te.index)
        assert not dup, f"Duplicate rows across splits: {len(dup)}"

        # simple leakage: identical columns to target
        leaks = [c for c in tr.columns if c != self.target
                 and tr[c].equals(tr[self.target])]
        assert not leaks, f"Potential leakage columns {leaks}"

    # ----------------------------- 5·4
    def freeze_preprocessor(self) -> None:
        num_cols = self.df.select_dtypes("number").columns
        scaler = Pipeline([("scale", StandardScaler())])
        scaler.fit(self.df[num_cols])

        joblib.dump(scaler, self.MODEL/"preprocessor.joblib")

        # checksum for manifest
        sha = hashlib.sha256(
            Path(self.MODEL/"preprocessor.joblib").read_bytes()).hexdigest()[:12]
        meta = {"path": "models/preprocessor.joblib", "sha256": sha}
        (self.MODEL/"preprocessor_manifest.json").write_text(json.dumps(meta, indent=2))

    # ----------------------------- Orchestrator
    def run(self):
        tr, val, te = self.split_data()
        self.build_baseline(tr, te)
        self.sanity_checks()
        self.freeze_preprocessor()
        print("🟢 Split + baseline phase complete")

# ------------------------------------------------------------------


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--oversample", action="store_true",
                   help="apply SMOTE to training fold")
    SplitAndBaseline(**vars(p.parse_args())).run()


if __name__ == "__main__":
    _cli()
