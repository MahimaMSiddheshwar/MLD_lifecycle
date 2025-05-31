"""
split_and_baseline.py ─ Phase 5·½
---------------------------------
Consumes the frozen splits produced in Phase 4·½,
calculates a “beat-that” baseline, runs sanity checks,
and serialises the *final* preprocessing object.

    python -m Data_Cleaning.split_and_baseline \
           --target is_churn --oversample
"""

from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# ──────────────────────────────────────────────────────────────
SPLIT_DIR = Path("data/splits")
PROC_PARQUET = Path("data/processed/scaled.parquet")
BASELINE_DIR = Path("reports/baseline")
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────


class baslineModel:
    """Phase 5·½ – baseline & pipeline-freeze (no re-splitting)"""

    def __init__(self, target: str, oversample: bool = False):
        self.target = target
        self.oversample = oversample

        self.train, self.val, self.test = self._load_frozen()

    # ---------- 5·0 Verify frozen split -----------------------
    @staticmethod
    def _checksum(fp: Path) -> str:
        return hashlib.sha256(fp.read_bytes()).hexdigest()[:12]

    def _load_frozen(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        for fn in ["train.parquet", "val.parquet", "test.parquet"]:
            assert (
                SPLIT_DIR/fn).exists(), f"missing {fn} – run Phase 4·½ first"

        # integrity check – hashes must match manifest
        manifest_path = SPLIT_DIR/"split_manifest.json"
        meta: Dict = json.loads(manifest_path.read_text())
        for fn in ["train.parquet", "val.parquet", "test.parquet"]:
            hash_now = self._checksum(SPLIT_DIR/fn)
            hash_manifest = meta["hashes"][fn]
            assert hash_now == hash_manifest, f"{fn} hash mismatch – do *not* re-split!"

        tr = pd.read_parquet(SPLIT_DIR/"train.parquet")
        va = pd.read_parquet(SPLIT_DIR/"val.parquet")
        te = pd.read_parquet(SPLIT_DIR/"test.parquet")

        if self.oversample and tr[self.target].dtype.kind not in "if":
            X_tr, y_tr = tr.drop(columns=self.target), tr[self.target]
            X_tr, y_tr = SMOTE(random_state=0).fit_resample(X_tr, y_tr)
            tr = pd.concat([X_tr, y_tr], axis=1)

        return tr, va, te

    # ---------- 5·1 Baseline model ----------------------------
    def baseline(self):
        y_test = self.test[self.target]

        if y_test.dtype.kind in "if":                         # regression
            y_pred = np.full_like(
                y_test, self.train[self.target].mean(), dtype=float)
            metrics = {
                "type": "mean_regressor",
                "mae": mean_absolute_error(y_test, y_pred),
                "r2":  r2_score(y_test, y_pred)
            }
        else:                                                # classification
            majority = self.train[self.target].mode()[0]
            y_pred = np.full_like(y_test, majority)
            metrics = {
                "type": "majority_class",
                "majority_class": int(majority),
                "accuracy": accuracy_score(y_test, y_pred),
                "f1":       f1_score(y_test, y_pred, zero_division=0)
            }

        (BASELINE_DIR/"baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------- 5·2 Sanity checks -----------------------------
    def sanity(self):
        dup_idx = set(self.train.index) & set(self.test.index)
        assert not dup_idx, f"duplicates across splits: {len(dup_idx)}"

        leaks = [c for c in self.train.columns if c != self.target
                 and self.train[c].equals(self.train[self.target])]
        assert not leaks, f"perfect-copy leakage columns: {leaks}"

    # ---------- 5·3 Freeze lightweight pre-processor ----------
    def freeze_preprocessor(self):
        num_cols = self.train.select_dtypes("number").columns
        scaler = Pipeline([("std", StandardScaler())]
                          ).fit(self.train[num_cols])

        joblib.dump(scaler, MODEL_DIR/"preprocessor.joblib")
        sha = baslineModel._checksum(MODEL_DIR/"preprocessor.joblib")
        (MODEL_DIR/"preprocessor_manifest.json").write_text(
            json.dumps({"sha256": sha,
                        "saved": datetime.utcnow().isoformat(timespec="seconds")},
                       indent=2)
        )

    # ---------- Orchestrator ----------------------------------
    def run(self):
        self.baseline()
        self.sanity()
        self.freeze_preprocessor()
        print("✅  Phase 5·½ complete – baseline & pre-processor ready.")


# ──────────────────────────────────────────────────────────────
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="target column name")
    ap.add_argument("--oversample", action="store_true",
                    help="SMOTE on train fold")
    baslineModel(**vars(ap.parse_args())).run()


if __name__ == "__main__":
    _cli()
