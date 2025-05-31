"""
Feature-selection + deterministic split  (Phase 4·½)
----------------------------------------------------
➜ python -m Data_Analysis.feature_selector --target is_churn
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("feat_sel")

RAW = Path("data/interim/clean.parquet")
SPLIT = Path("data/splits")
SPLIT.mkdir(parents=True, exist_ok=True)


def low_variance_drop(X, thr=1e-4):
    vt = VarianceThreshold(thr)
    vt.fit(X)
    return X.columns[~vt.get_support()]


def mi_drop(X, y, thr=0.0):
    mi = mutual_info_classif(X, y, discrete_features="auto")
    return X.columns[mi <= thr]


def corr_prune(X, rho=0.95):
    corr = X.corr().abs()
    drop = set()
    for col in corr:
        if col in drop:
            continue
        high = corr.index[(corr[col] > rho) & (corr.index != col)]
        drop.update(high)
    return list(drop)


def main(args):
    df = pd.read_parquet(RAW)
    X, y = df.drop(columns=[args.target]), df[args.target]

    # ── split early ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test,  y_test], axis=1)
    train.to_parquet(SPLIT/"train.parquet", index=False)
    test .to_parquet(SPLIT/"test.parquet",  index=False)
    log.info("split saved: train=%s  test=%s", train.shape, test.shape)

    # ── feature filters (train-only) ──────────────────────────
    drop = set()

    drop |= set(low_variance_drop(X_train, args.var_thr))
    drop |= set(mi_drop(X_train, y_train, args.mi_thresh))
    drop |= set(corr_prune(X_train, args.corr_thresh))

    keep = [c for c in X_train.columns if c not in drop]
    Path("reports/feature_selection").mkdir(parents=True, exist_ok=True)
    (Path("reports/feature_selection")/"feature_plan.json").write_text(
        json.dumps({"keep": keep, "drop": sorted(drop)}, indent=2))

    log.info("feature plan saved – kept=%d, dropped=%d", len(keep), len(drop))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--var-thr",    type=float, default=1e-4)
    p.add_argument("--mi-thresh",  type=float, default=0.0)
    p.add_argument("--corr-thresh", type=float, default=0.95)
    args = p.parse_args()
    main(args)
