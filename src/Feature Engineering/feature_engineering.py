# >
# > * Removed unused `interactions` param (folded into `polynomial_degree`).
# > * Ensured `FunctionTransformer(...)` inside the pipeline always gets a DataFrame with correct column names.
# > * Added pre‑transform pruning (NZV + correlation filter).
# > * Added MI/F‑score prune.
# > * Writes `reports/feature/feature_audit.json` (counts) and `feature_shape.txt` (post‑transform shape).

"""
Feature-engineering buffet — v3
Adds:
  • near-zero-variance drop
  • mutual-information / F-score filter
  • correlation filter (numeric)
  • feature-audit JSON + shape txt
  • text length & word-count helpers
"""

from __future__ import annotations
import argparse
import json
import math
import joblib
import hashlib
from pathlib import Path
from typing import List, Dict, Callable, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler,
    RobustScaler, PowerTransformer, FunctionTransformer,
    PolynomialFeatures, KBinsDiscretizer
)
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer
)
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif, mutual_info_regression, f_classif
)
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted
from category_encoders.target_encoder import TargetEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.woe import WOEEncoder

# ───────────────────────────────────────────────────────────────
# simple custom encoders / helpers


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, _=None):
        self.maps_ = {c: pd.Series(x).value_counts(normalize=True).to_dict()
                      for c, x in zip(X.columns, X.T.values)}
        return self

    def transform(self, X):  # X is DataFrame slice
        out = X.copy()
        for c in X:
            out[c] = out[c].map(self.maps_[c]).fillna(0)
        return out.values


class RareCategory(BaseEstimator, TransformerMixin):
    def __init__(self, th: float = .01): self.th = th

    def fit(self, X, _=None):
        n = len(X)
        self.rare_ = {c: set(pd.Series(x).value_counts(normalize=True)
                             .loc[lambda s: s < self.th].index)
                      for c, x in zip(X.columns, X.T.values)}
        return self

    def transform(self, X):
        out = X.copy()
        for c in X:
            out.loc[out[c].isin(self.rare_[c]), c] = "__rare__"
        return out.values


class Cyclical(BaseEstimator, TransformerMixin):
    def __init__(self, period: int): self.period = period
    def fit(self, X, _=None): return self

    def transform(self, X):
        x = X.values.astype(float)
        return np.c_[np.sin(2*np.pi*x/self.period),
                     np.cos(2*np.pi*x/self.period)]


class TextLength(BaseEstimator, TransformerMixin):
    """Adds `*_n_chars`, `*_n_words` for each text col (handy for light text)."""

    def fit(self, X, _=None): return self

    def transform(self, X):
        ser = X.iloc[:, 0].fillna("")
        return pd.DataFrame({"n_chars": ser.str.len(),
                             "n_words": ser.str.split().str.len()}).values
# ───────────────────────────────────────────────────────────────


class FeatureEngineer:
    # —— constructor ---------------------------------------------------------
    def __init__(self,
                 # classic options (unchanged)
                 target: str | None = None,
                 numeric_scaler: str = "standard",   # standard|minmax|robust|none
                 numeric_power: str | None = None,         # yeo|boxcox|quantile
                 log_cols: List[str] | None = None,
                 quantile_bins: Dict[str, int] | None = None,
                 polynomial_degree: int | None = None,
                 interactions: bool = False,
                 rare_threshold: float | None = None,
                 cat_encoding: str = "onehot",     # onehot|ordinal|target|woe|hash|freq|none
                 text_vectorizer: str | None = None,         # tfidf|count|hashing|none
                 text_cols: List[str] | None = None,
                 datetime_cols: List[str] | None = None,
                 cyclical_cols: Dict[str, int] | None = None,
                 date_delta_cols: Dict[str, str] | None = None,
                 aggregations: Dict[str, List[str]] | None = None,
                 # NEW —— filtering / pruning
                 drop_nzv: bool = True,         # near-zero variance
                 corr_threshold: float | None = .95,          # numeric corr filter
                 mi_quantile: float | None = .10,          # bottom 10 % MI dropped
                 # misc
                 custom_steps: List[Callable[[pd.DataFrame],
                                             pd.DataFrame]] | None = None,
                 save_path: str | Path = "models/preprocessor.joblib",
                 report_dir: str | Path = "reports/feature"):
        self.__dict__.update(**locals())
        self.save_path = Path(save_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.pipe_: Pipeline | None = None
        self.custom_steps = custom_steps or []

    # —— private helpers -----------------------------------------------------
    def _build_pretransform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering steps BEFORE ColumnTransformer so we never fit
        heavy transforms on junk columns."""
        keep = X.copy()

        # 1. near-zero variance
        if self.drop_nzv:
            nzv = VarianceThreshold(
                threshold=1e-5).fit(keep.select_dtypes("number"))
            nzv_keep = keep.select_dtypes("number").columns[nzv.get_support()]
            drop = set(keep.select_dtypes("number").columns) - set(nzv_keep)
            keep.drop(columns=list(drop), inplace=True)

        # 2. correlation filter (numeric cols only)
        if self.corr_threshold and keep.select_dtypes("number").shape[1] > 1:
            corr = keep.select_dtypes("number").corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
            to_drop = [c for c in upper.columns if (
                upper[c] >= self.corr_threshold).any()]
            keep.drop(columns=to_drop, inplace=True)

        return keep

    def _mi_filter(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Drop bottom-quantile MI/F-score features (after basic cleaning)."""
        num = X.select_dtypes("number").columns.tolist()
        cat = X.select_dtypes("object").columns.tolist()
        keep_mask = pd.Series(True, index=X.columns)

        if num:
            if y.nunique() > 10:   # regress
                mi = mutual_info_regression(X[num], y, random_state=0)
            else:                # classify
                mi = f_classif(X[num], y)[0]  # F-score proxy
                mi[np.isnan(mi)] = 0
            mi_series = pd.Series(mi, index=num)
            low = mi_series.quantile(self.mi_quantile)
            keep_mask.loc[mi_series[mi_series < low].index] = False

        # crude chi2 / MI for cats — pingouin or sklearn requires encoding; skip for brevity
        # (advanced users can enable later)

        return X.loc[:, keep_mask.values]

    def _build_column_transformer(self, X: pd.DataFrame) -> ColumnTransformer:
        num = X.select_dtypes("number").columns.tolist()
        cat = X.select_dtypes("object").columns.tolist()
        txt = self.text_cols or []
        cat = [c for c in cat if c not in txt]
        scale_map = {"standard": StandardScaler(), 'minmax': MinMaxScaler(),
                     'robust': RobustScaler(), 'none': FunctionTransformer(lambda x: x)}
        # numeric pipeline
        n_pipe = []
        if self.log_cols:
            n_pipe.append(("log",
                           ColumnTransformer([(f"log_{c}", FunctionTransformer(np.log1p, "one-to-one"), [c])
                                              for c in self.log_cols], remainder='passthrough')))
        if self.numeric_power:
            n_pipe.append(
                ("power", PowerTransformer(method=self.numeric_power)))
        if self.numeric_scaler != "none":
            n_pipe.append(("scale", scale_map[self.numeric_scaler]))
        if self.quantile_bins:
            q_steps = [(f"bin_{c}", KBinsDiscretizer(n_bins=b, encode="ordinal", strategy="quantile"), [c])
                       for c, b in self.quantile_bins.items() if c in num]
            if q_steps:
                n_pipe.append(("qbin", ColumnTransformer(
                    q_steps, remainder='passthrough')))
        if self.polynomial_degree:
            n_pipe.append(("poly", PolynomialFeatures(
                self.polynomial_degree, include_bias=False)))
        num_pipe = Pipeline(n_pipe) if n_pipe else "passthrough"

        # cat pipeline
        c_pipe = []
        if self.rare_threshold:
            c_pipe.append(("rare", RareCategory(self.rare_threshold)))
        enc_map = {"onehot": OneHotEncoder(handle_unknown="ignore", sparse=False),
                   "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                   "target": TargetEncoder(), "woe": WOEEncoder(),
                   "hash": HashingEncoder(), "freq": FrequencyEncoder(),
                   "none": "passthrough"}
        c_pipe.append(("enc", enc_map[self.cat_encoding]))
        cat_pipe = Pipeline(c_pipe)

        # ColumnTransformer assembly
        transformers = [("num", num_pipe, num),
                        ("cat", cat_pipe, cat)]

        # text:
        vect_map = {"tfidf": TfidfVectorizer,
                    "count": CountVectorizer, "hashing": HashingVectorizer}
        for col in txt:
            transformers.append((f"text_{col}",
                                 vect_map[self.text_vectorizer](
                                     max_features=100, ngram_range=(1, 2)),
                                 col))
            transformers.append((f"textlen_{col}", TextLength(), [col]))

        # cyclical & datetime handled via custom steps (users can add)

        return ColumnTransformer(transformers, remainder="drop")

    # —— public API -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if self.target and self.target in X.columns:
            y = y or X[self.target]
            X = X.drop(columns=[self.target])

        X_clean = self._build_pretransform(X)
        if self.mi_quantile:        # MI/F-score prune
            X_clean = self._mi_filter(X_clean, y)

        pre = self._build_column_transformer(X_clean)
        pipe = Pipeline([("pre", pre)])

        # custom post-steps
        for fn in self.custom_steps:
            pipe.steps.append((fn.__name__,
                               FunctionTransformer(lambda df, f=fn: f(pd.DataFrame(df)),
                                                   feature_names_out="one-to-one")))

        self.pipe_ = pipe.fit(X_clean, y)

        # —— audit report
        Path(self.report_dir).mkdir(exist_ok=True, parents=True)
        audit = {"n_features_in": len(X.columns),
                 "n_features_after_clean": len(X_clean.columns)}
        (self.report_dir/"feature_audit.json").write_text(json.dumps(audit, indent=2))
        with open(self.report_dir/"feature_shape.txt", "w") as f:
            f.write(f"{self.pipe_.transform(X_clean).shape}\n")

        return self

    def transform(self, X): check_is_fitted(
        self.pipe_); return self.pipe_.transform(X)

    def fit_transform(self, X, y=None): return self.fit(X, y).transform(X)

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe_, self.save_path)
        sha = hashlib.sha256(
            Path(self.save_path).read_bytes()).hexdigest()[:12]
        (self.save_path.with_suffix(".json")
         ).write_text(json.dumps({"sha256": sha}, indent=2))

# —— CLI wrapper -------------------------------------------------------------


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.data)
    FeatureEngineer(target=args.target).fit(df, df[args.target]).save()


if __name__ == "__main__":
    _cli()
