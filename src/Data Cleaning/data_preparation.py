"""
src/ml_pipeline/prepare.py
-------------------------------------------------
End-to-end data-preparation pipeline for Phase-3.
Run standalone:

    python -m ml_pipeline.prepare        # defaults
    python -m ml_pipeline.prepare --knn  # fancy imputation
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pandera as pa
import pyjanitor as jan
import missingno as msno
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer
)
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# ───────── logging ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prepare")

# ───────── paths & schema  ────────────────────────────────────
RAW = Path("data/raw/combined.parquet")
INT = Path("data/interim/clean.parquet")
PROC = Path("data/processed/scaled.parquet")

schema = pa.DataFrameSchema({
    "uid": pa.Column(pa.String, nullable=False),
    "age": pa.Column(pa.Int, checks=pa.Check.ge(13)),
    "city": pa.Column(pa.String),
    "last_login": pa.Column(pa.DateTime),
    "amount": pa.Column(pa.Float),
    "is_churn": pa.Column(pa.Int, checks=pa.Check.isin([0, 1]))
})

# ══════════════════════════════════════════════════════════════


class DataPreparer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.df: pd.DataFrame | None = None

    # ── 3A Schema & coercion ──────────────────────────────────
    def load_and_validate(self):
        self.df = pd.read_parquet(RAW).janitor.clean_names()
        self.df = schema.validate(self.df)
        log.info("validated: %s", self.df.shape)

    # ── 3B Missing-value strategy ─────────────────────────────
    def impute(self):
        df = self.df
        # visual snapshot
        msno.matrix(df.sample(min(1000, len(df))))
        # numeric → median / KNN
        num_cols = df.select_dtypes("number").columns
        cat_cols = df.select_dtypes("object").columns

        if self.cfg["knn_impute"]:
            imputer = KNNImputer(n_neighbors=5)
            df[num_cols] = imputer.fit_transform(df[num_cols])
        else:
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # categorical → mode
        for c in cat_cols:
            df[c].fillna(df[c].mode(dropna=True)[0], inplace=True)

        self.df = df
        log.info("imputed: nulls=%d", self.df.isna().sum().sum())

    # ── 3C Outlier handling (IQR / z-score / IF) ──────────────
    def treat_outliers(self):
        df, method = self.df, self.cfg["outlier"]
        if method == "iqr":
            q1, q3 = df.amount.quantile([.25, .75])
            fence = 1.5*(q3-q1)
            df = df[df.amount.between(q1-fence, q3+fence)]
        elif method == "zscore":
            z = np.abs(stats.zscore(df.amount))
            df = df[z < 3]
        elif method == "iso":
            iso = IsolationForest(contamination=0.01, random_state=7)
            mask = iso.fit_predict(df[["amount"]]) == 1
            df = df[mask]
        self.df = df
        log.info("outlier-treated: %s", df.shape)

    # ── 3D Transformation & scaling ───────────────────────────
    def transform(self):
        df = self.df.copy()
        num = df.select_dtypes("number").columns
        scaler_name = self.cfg["scaler"]

        # power transform first (log/yeo)
        if self.cfg["log_amount"]:
            df["amount"] = np.log1p(df["amount"])

        if scaler_name == "standard":
            scaler = StandardScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:  # yeo-johnson
            scaler = PowerTransformer(method="yeo-johnson")

        df[num] = scaler.fit_transform(df[num])
        self.df = df
        log.info("scaled with %s", scaler_name)

    # ── 3E Class-imbalance remed. (optional) ──────────────────
    def rebalance(self):
        if not self.cfg["balance"]:
            return
        X = self.df.drop(columns="is_churn")
        y = self.df["is_churn"]

        if self.cfg["balance"] == "smote":
            X_bal, y_bal = SMOTE(random_state=0).fit_resample(X, y)
        else:
            X_bal, y_bal = NearMiss().fit_resample(X, y)

        self.df = pd.concat([X_bal, y_bal], axis=1)
        log.info("rebalance → %s (pos=%0.2f)",
                 self.df.shape, self.df.is_churn.mean())

    # ── 3F Versioning & lineage manifest ─────────────────────
    def save(self):
        INT.parent.mkdir(parents=True, exist_ok=True)
        PROC.parent.mkdir(parents=True, exist_ok=True)

        # interim save (pre-scaling)
        self.df.to_parquet(INT, index=False)

        # final save
        self.df.to_parquet(PROC, index=False)

        meta = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "rows": len(self.df),
            "scaler": self.cfg["scaler"],
            "outlier": self.cfg["outlier"],
            "balance": self.cfg["balance"],
            "raw_sha": self.cfg.get("raw_sha", "n/a")
        }
        Path("reports/lineage").mkdir(parents=True, exist_ok=True)
        (Path("reports/lineage") / "prep_manifest.json").write_text(
            json.dumps(meta, indent=2)
        )
        log.info("✅ saved interim & processed; lineage manifest written")

    # ── pipeline orchestration ────────────────────────────────
    def run(self):
        self.load_and_validate()
        self.impute()
        self.treat_outliers()
        self.transform()
        self.rebalance()
        self.save()


# ══════════════════════════════════════════════════════════════
def cli():
    p = argparse.ArgumentParser(prog="prepare")
    p.add_argument("--knn", action="store_true",
                   help="use KNNImputer instead of median/mode")
    p.add_argument("--outlier", choices=["iqr", "zscore", "iso"],
                   default="iqr")
    p.add_argument("--scaler", choices=["standard", "robust", "yeo"],
                   default="standard")
    p.add_argument("--balance", choices=["none", "smote", "nearmiss"],
                   default="none")
    cfg = vars(p.parse_args())

    cfg = {  # defaults + flags
        "knn_impute": cfg["knn"],
        "outlier": cfg["outlier"],
        "log_amount": True,
        "scaler": cfg["scaler"],
        "balance": cfg["balance"] if cfg["balance"] != "none" else None
    }
    DataPreparer(cfg).run()


if __name__ == "__main__":
    cli()
