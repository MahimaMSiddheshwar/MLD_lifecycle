"""
feature_select_split.py  – Phase 4·½
────────────────────────────────────
Refines the feature matrix **and** freezes an
immutable Train / Val / Test split so every

"""

from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression
)

# input (pre-scaling, no leakage)
RAW_INT = Path("data/interim/clean.parquet")
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path("reports/feature_select")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════


def sha12(fp: Path) -> str:
    return hashlib.sha256(fp.read_bytes()).hexdigest()[:12]


def near_zero_var(df: pd.DataFrame, thresh: float = .999) -> List[str]:
    return [c for c in df.columns if df[c].value_counts(normalize=True).values[0] > thresh]


def high_null(df: pd.DataFrame, thresh: float) -> List[str]:
    return [c for c in df.columns if df[c].isna().mean() > thresh]


def high_corr(df: pd.DataFrame, thresh: float) -> List[str]:
    corr = df.select_dtypes("number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
    return [column for column in upper.columns if any(upper[column] > thresh)]


def low_importance(df: pd.DataFrame,
                   target: str,
                   is_classif: bool,
                   k: int = 0,
                   mi_thresh: float | None = None) -> List[str]:
    X = df.drop(columns=[target])
    y = df[target]
    num = X.select_dtypes("number").columns
    cat = X.select_dtypes("object").columns

    # encode categoricals as codes for quick MI/F-score
    X_enc = X.copy()
    for c in cat:
        X_enc[c] = X_enc[c].astype("category").cat.codes

    if is_classif:
        mi = mutual_info_classif(
            X_enc, y, discrete_features="auto", random_state=0)
        f_val, _ = f_classif(X_enc, y)
    else:
        mi = mutual_info_regression(X_enc, y, random_state=0)
        f_val, _ = f_regression(X_enc, y)

    imp = pd.DataFrame({"feat": X.columns, "mi": mi, "f": f_val})
    if mi_thresh is not None:
        return imp.loc[imp.mi < mi_thresh, "feat"].tolist()
    else:
        # keep top-k if k>0
        return imp.sort_values("mi").head(k)["feat"].tolist() if k else []


def main(cfg: Dict):
    df = pd.read_parquet(RAW_INT)
    drops: Dict[str, List[str]] = {}

    # ---------- 1  MANUAL DROP --------------------------------
    if cfg["manual_drop"]:
        manual = [c.strip() for c in cfg["manual_drop"].split(",")
                  if c.strip() in df.columns]
        df = df.drop(columns=manual)
        drops["manual"] = manual

    # ---------- 2  AUTO FILTERS -------------------------------
    nzv = near_zero_var(df, cfg["nzv_thresh"])
    nulls = high_null(df, cfg["null_thresh"])
    leaks = cfg["leak_cols"]

    df = df.drop(columns=set(nzv + nulls + leaks))
    drops.update({"near_zero_var": nzv, "high_null": nulls, "leak": leaks})

    corr_drop = high_corr(df, cfg["corr_thresh"])
    df = df.drop(columns=corr_drop)
    drops["high_corr"] = corr_drop

    imp_drop = low_importance(df, cfg["target"],
                              cfg["task"] == "cls",
                              mi_thresh=cfg["mi_thresh"])
    df = df.drop(columns=imp_drop)
    drops["low_importance"] = imp_drop

    Path(REPORT_DIR/"features_dropped.json").write_text(json.dumps(drops, indent=2))

    # ---------- 3  SPLIT  ------------------------------------
    strat = df[cfg["target"]] if (
        cfg["task"] == "cls" and cfg["stratify"]) else None
    train, test = train_test_split(df, test_size=cfg["test_size"],
                                   random_state=cfg["seed"], stratify=strat)
    val_size_adj = cfg["val_size"] / (1 - cfg["test_size"])
    strat_temp = strat.loc[train.index] if strat is not None else None
    train, val = train_test_split(train, test_size=val_size_adj,
                                  random_state=cfg["seed"], stratify=strat_temp)

    (SPLIT_DIR/"train.parquet").write_bytes(train.to_parquet(index=False))
    (SPLIT_DIR/"val.parquet").write_bytes(val.to_parquet(index=False))
    (SPLIT_DIR/"test.parquet").write_bytes(test.to_parquet(index=False))

    manifest = {
        "timestamp": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "rows": {k: len(v) for k, v in
                 dict(train=train, val=val, test=test).items()},
        "stratify": cfg["stratify"],
        "hashes": {fn: sha12(SPLIT_DIR/f"{fn}.parquet")
                   for fn in ["train", "val", "test"]}
    }
    (SPLIT_DIR/"split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("🟢 Phase 4·½ complete – feature list frozen & splits saved.")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--task", choices=["cls", "reg"], required=True,
                    help="classification or regression")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify", action="store_true")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size",  type=float, default=0.1)
    ap.add_argument("--manual_drop", default="",
                    help="comma-separated feature names to drop")
    ap.add_argument("--leak_cols", default="",
                    help="comma-separated leaky cols")
    ap.add_argument("--null_thresh", type=float, default=0.5,
                    help="drop cols with >X null-ratio")
    ap.add_argument("--nzv_thresh",  type=float, default=0.999,
                    help="drop cols where most common value ≥ X")
    ap.add_argument("--corr_thresh", type=float, default=0.9,
                    help="drop second of pairs with corr ≥ X")
    ap.add_argument("--mi_thresh",   type=float, default=None,
                    help="mutual-info / F-score threshold (drop below)")
    args = vars(ap.parse_args())

    main(args)
