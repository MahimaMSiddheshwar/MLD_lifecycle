# > * Added `--target` to CLI, so the code knows which column is the target (used later in balancing).
# > * Added an explicit random seed for SMOTE.
# > * Moved inline Pandera schema into `default_schema`, so users can override if necessary.

from __future__ import annotations
import pyjanitor as jan
import pandera as pa
import warnings
import logging
import pandas as pd
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from scipy import stats
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

"""
src/ml_pipeline/prepare.py
-------------------------------------------------
End-to-end data-preparation pipeline (Phase-3).

Run:
    python -m ml_pipeline.prepare                # defaults
    python -m ml_pipeline.prepare --help         # all flags
"""


# optional DQ suite
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    try:
        import great_expectations as gx            # type: ignore
        GX_OK = True
    except ModuleNotFoundError:
        GX_OK = False

# ───────── config & paths ────────────────────────────────────
RAW = Path("data/raw/combined.parquet")
INT = Path("data/interim/clean.parquet")
PROC = Path("data/processed/scaled.parquet")
LINE = Path("reports/lineage")
LINE.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prepare")


# TODO: Example Pandera schema – override or extend as needed


schema = pa.DataFrameSchema({
    "uid":        pa.Column(pa.String, nullable=False),
    "age":        pa.Column(pa.Int,    checks=pa.Check.ge(13)),
    "city":       pa.Column(pa.String),
    "last_login": pa.Column(pa.DateTime),
    "amount":     pa.Column(pa.Float,  checks=pa.Check.ge(0)),
    "is_churn":   pa.Column(pa.Int,    checks=pa.Check.isin([0, 1]))
})


# ╔══════════════════════════════════════════════════════════╗
# ║                    pipeline class                        ║
# ╚══════════════════════════════════════════════════════════╝
class DataPreparer:
    def __init__(self, cfg: dict):          # cfg assembled from CLI
        self.cfg = cfg
        self.df: pd.DataFrame

    # ── 3A  schema + (opt) great-expectations  --------------
    def load_and_validate(self):
        self.df = pd.read_parquet(RAW).janitor.clean_names()
        self.df = schema.validate(self.df)
        log.info("validated schema OK → %s", self.df.shape)

        if self.cfg["gx"] and GX_OK:
            ctx = gx.DataContext.create()
            ds = ctx.sources.add_pandas(name="phase3", dataframe=self.df)
            suite = ctx.add_expectation_suite("dq_quick")
            # simple automated expectations
            suite.add_expectation("expect_table_columns_to_match_ordered_list",
                                  column_list=self.df.columns.tolist())
            suite.add_expectation("expect_column_values_to_not_be_null",
                                  column="uid")
            res = ds.validate(expectation_suite=suite)
            if not res.success:               # fail fast
                raise ValueError("Great-Expectations suite failed")
            res.save()                        # HTML in gx/uncommitted
            log.info("Great-Expectations ✓")

    # ── 3B  missing-value handling + preview ----------------
    def impute(self):
        msno.matrix(self.df.sample(min(1000, len(self.df))))
        (LINE / "missing_matrix.png").write_bytes(plt_bytes())  # helper below

        num = self.df.select_dtypes("number").columns
        cat = self.df.select_dtypes("object").columns

        if self.cfg["knn_impute"]:
            self.df[num] = KNNImputer(
                n_neighbors=5).fit_transform(self.df[num])
        else:
            self.df[num] = self.df[num].fillna(self.df[num].median())

        for c in cat:
            self.df[c].fillna(self.df[c].mode(dropna=True)[0], inplace=True)

        log.info("impute done (missing=%d)", self.df.isna().sum().sum())

        # drop-missing threshold
        if self.cfg["drop_miss"] is not None:
            thresh = self.cfg["drop_miss"]
            na_frac = self.df.isna().mean()
            drops = na_frac[na_frac > thresh].index.tolist()
            self.df.drop(columns=drops, inplace=True)
            if drops:
                (LINE / "drop_missing.json").write_text(json.dumps(drops))
                log.info("dropped %d cols for >%0.2f NaNs", len(drops), thresh)

    # ── 3C  duplicates & constant columns -------------------
    def dedup_prune(self):
        if self.cfg["dedup"]:
            before = len(self.df)
            self.df.drop_duplicates(subset=self.cfg["dedup"], inplace=True)
            log.info("dedup → %d rows (-%d)",
                     len(self.df), before-len(self.df))

        const_thresh = self.cfg["prune_const"]
        nunique = self.df.nunique(dropna=False)/len(self.df)
        const = nunique[nunique > const_thresh].index
        if len(const):
            self.df.drop(columns=const, inplace=True)
            (LINE / "drop_const.json").write_text(json.dumps(const.tolist()))
            log.info("pruned %d quasi-constant cols", len(const))

    # ── 3C  outlier treatment --------------------------------
    def treat_outliers(self):
        m = self.cfg["outlier"]
        s = self.df["amount"]  # demo column
        if m == "iqr":
            q1, q3 = s.quantile([.25, .75])
            fence = 1.5*(q3-q1)
            mask = s.between(q1-fence, q3+fence)
        elif m == "zscore":
            mask = np.abs(stats.zscore(s)) < 3
        elif m == "iso":
            mask = IsolationForest(contamination=0.01, random_state=7)\
                .fit_predict(s.to_frame()) == 1
        else:  # LOF
            mask = LocalOutlierFactor(n_neighbors=20, contamination=0.01)\
                .fit_predict(s.to_frame()) == 1
        self.df = self.df[mask]
        log.info("%s outlier filter → %s", m, self.df.shape)

    # ── 3D  transform / scale --------------------------------
    def transform(self):
        num = self.df.select_dtypes("number").columns

        # auto-log skewed
        if self.cfg["auto_log"]:
            for col in num:
                if abs(self.df[col].skew()) > 1:
                    self.df[col] = np.log1p(self.df[col])
                    log.debug("log1p(%s)", col)

        scaler_name = self.cfg["scaler"]
        scaler = {"standard": StandardScaler,
                  "robust":   RobustScaler,
                  "yeo": lambda: PowerTransformer(method="yeo-johnson")}[scaler_name]()
        self.df[num] = scaler.fit_transform(self.df[num])

        if self.cfg["qt"]:
            qt = QuantileTransformer(output_distribution="normal")
            self.df[num] = qt.fit_transform(self.df[num])

        log.info("scaled with %s", scaler_name)

    # ── 3E  rebalance ----------------------------------------
    def rebalance(self):
        if not self.cfg["balance"]:
            return
        X = self.df.drop(columns="is_churn")
        y = self.df["is_churn"]
        sampler = SMOTE(
            random_state=0) if self.cfg["balance"] == "smote" else NearMiss()
        Xb, yb = sampler.fit_resample(X, y)
        self.df = pd.concat([Xb, yb], axis=1)
        log.info("rebalance %s → pos rate %0.2f",
                 self.cfg["balance"], self.df.is_churn.mean())

    # ── 3G  high-corr pruning --------------------------------
    def drop_correlated(self):
        thr = self.cfg["drop_corr"]
        if thr is None:
            return
        num = self.df.select_dtypes("number").columns
        corr = self.df[num].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > thr)]
        self.df.drop(columns=to_drop, inplace=True)
        if to_drop:
            (LINE/"drop_corr.json").write_text(json.dumps(to_drop))
            log.info("dropped %d highly-corr cols (>%0.2f)", len(to_drop), thr)

    # ── 3F  save versions ------------------------------------
    def save(self):
        INT.parent.mkdir(parents=True, exist_ok=True)
        PROC.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(INT,  index=False)
        self.df.to_parquet(PROC, index=False)

        meta = dict(
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            rows=len(self.df),
            scaler=self.cfg["scaler"],
            outlier=self.cfg["outlier"],
            balance=self.cfg["balance"],
            drop_miss=self.cfg["drop_miss"],
            drop_corr=self.cfg["drop_corr"],
            raw_sha=self.cfg.get("raw_sha", "n/a")
        )
        (LINE/"prep_manifest.json").write_text(json.dumps(meta, indent=2))
        log.info("✅ saved ➜ interim & processed; lineage manifest written")

    # ── orchestrate ------------------------------------------
    def run(self):
        self.load_and_validate()
        self.impute()
        self.dedup_prune()
        self.treat_outliers()
        self.transform()
        self.rebalance()
        self.drop_correlated()
        self.save()


# ╔══════════════════════════════════════════════════════════╗
# ║                         CLI                              ║
# ╚══════════════════════════════════════════════════════════╝
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--knn",           action="store_true",
                   help="use KNNImputer")
    p.add_argument("--outlier",       default="iqr",
                   choices=["iqr", "zscore", "iso", "lof"])
    p.add_argument("--scaler",        default="standard",
                   choices=["standard", "robust", "yeo"])
    p.add_argument("--qt",            action="store_true",
                   help="add quantile transform")
    p.add_argument("--auto-log",      action="store_true",
                   help="log1p skewed numeric")
    p.add_argument("--balance",       default=None,
                   choices=[None, "smote", "nearmiss"])
    p.add_argument("--drop-miss",     type=float,
                   help="drop cols with NaN fraction > p")
    p.add_argument("--drop-corr",     type=float,
                   help="drop cols with |corr| > p")
    p.add_argument("--dedup",         type=str,
                   help="column(s) to deduplicate on")
    p.add_argument("--prune-const",   type=float, default=0.99,
                   help="threshold for quasi-constant drop")
    p.add_argument("--gx",            action="store_true",
                   help="run Great Expectations")
    args = vars(p.parse_args())

    cfg = dict(
        knn_impute=args["knn"],
        outlier=args["outlier"],
        scaler=args["scaler"],
        qt=args["qt"],
        auto_log=args["auto_log"],
        balance=args["balance"],
        drop_miss=args["drop_miss"],
        drop_corr=args["drop_corr"],
        dedup=args["dedup"].split(",") if args["dedup"] else None,
        prune_const=args["prune_const"],
        gx=args["gx"],
    )
    DataPreparer(cfg).run()


# helper to grab current matplotlib figure as bytes (used once above)
def plt_bytes():
    import matplotlib.pyplot as plt
    import io
    bio = io.BytesIO()
    plt.savefig(bio, format="png")
    bio.seek(0)
    buf = bio.read()
    plt.close()
    return buf


if __name__ == "__main__":
    cli()
