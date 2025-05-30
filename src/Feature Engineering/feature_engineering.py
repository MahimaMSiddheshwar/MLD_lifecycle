"""
Feature-engineering buffet — V2
* adds freq-encode, cyclical, interactions, quantile bins, date deltas,
  aggregation features, and a plug-in API.
"""

from __future__ import annotations
import argparse
import joblib
import math
import json
from pathlib import Path
from typing import List, Dict, Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    FunctionTransformer, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer)
from category_encoders.target_encoder import TargetEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.woe import WOEEncoder
from sklearn.preprocessing import KBinsDiscretizer

# ---------- helper blocks ----------------------------------------------------


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self): self.freq_: Dict[str, Dict] = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X:
            self.freq_[col] = X[col].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X:
            X[col] = X[col].map(self.freq_[col]).fillna(0)
        return X.values


class Cyclical(BaseEstimator, TransformerMixin):
    def __init__(self, period: int): self.period = period
    def fit(self, X, y=None): return self

    def transform(self, X):
        X = np.array(X).astype(float)
        return np.c_[np.sin(2*np.pi*X/self.period),
                     np.cos(2*np.pi*X/self.period)]


class DateDelta(BaseEstimator, TransformerMixin):
    def __init__(self, reference: str = "today"):
        self.ref = pd.Timestamp("now").normalize(
        ) if reference == "today" else pd.to_datetime(reference)

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = pd.to_datetime(X.iloc[:, 0])
        return (self.ref - X).dt.days.to_frame().values


class RareCategory(BaseEstimator, TransformerMixin):
    def __init__(self, th: float | int = .01): self.th = th; self.map = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for c in X:
            vc = X[c].value_counts(normalize=True)
            self.map[c] = set(vc[vc < self.th].index if isinstance(self.th, float)
                              else vc[vc < self.th/len(X)].index)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c, r in self.map.items():
            X.loc[X[c].isin(r), c] = "__rare__"
        return X.values
# -----------------------------------------------------------------------------


class FeatureEngineer:
    def __init__(
            self,
            target: str | None = None,
            numeric_scaler: str = "standard",          # standard|minmax|robust|none
            numeric_power: str | None = None,            # yeo|boxcox|quantile
            log_cols: List[str] | None = None,
            quantile_bins: Dict[str, int] | None = None,  # {"age":5}
            interactions: bool = False,
            rare_threshold: float | int | None = None,     # freq encode also ok
            # onehot|ordinal|target|woe|hash|freq|none
            cat_encoding: str = "onehot",
            text_vectorizer: str | None = None,          # tfidf|count|hashing
            text_cols: List[str] | None = None,
            datetime_cols: List[str] | None = None,      # raw cols to expand
            # {"dow":7,"month":12}
            cyclical_cols: Dict[str, int] | None = None,
            # {"signup_date":"2020-01-01"}
            date_delta_cols: Dict[str, str] | None = None,
            polynomial_degree: int | None = None,
            # {"customer_id":["amount_mean","amount_sum"]}
            aggregations: Dict[str, List[str]] | None = None,
            custom_steps: List[Callable[[pd.DataFrame],
                                        pd.DataFrame]] | None = None,
            save_path: str | Path = "models/preprocessor.joblib"):

        self.__dict__.update(**locals())
        self.save_path = Path(save_path)
        self.pipe_: Pipeline | None = None
        self.custom_steps = custom_steps or []

    # ---------------- internal helpers ---------------------------------------
    def _build(self, X: pd.DataFrame):
        num = X.select_dtypes("number").columns.tolist()
        cat = X.select_dtypes("object").columns.tolist()
        txt = self.text_cols or []
        cat = [c for c in cat if c not in txt]
        dt = self.datetime_cols or []
        cyc = self.cyclical_cols or {}
        delta = self.date_delta_cols or {}

        # numeric pipeline
        nsteps = []
        if self.log_cols:
            nsteps.append(("log", ColumnTransformer(
                [(f"log_{c}", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), [c])
                 for c in self.log_cols], remainder="passthrough")))
        if self.numeric_power:
            nsteps.append(
                ("power", PowerTransformer(method=self.numeric_power)))
        scale_map = {"standard": StandardScaler(), "minmax": MinMaxScaler(),
                     "robust": RobustScaler()}
        if self.numeric_scaler != "none":
            nsteps.append(("scale", scale_map[self.numeric_scaler]))
        if self.quantile_bins:
            qt = [(f"bin_{c}", KBinsDiscretizer(n_bins=b, encode="ordinal", strategy="quantile"), [c])
                  for c, b in self.quantile_bins.items()]
            nsteps.append(
                ("qbin", ColumnTransformer(qt, remainder="passthrough")))
        if self.polynomial_degree:
            nsteps.append(("poly", PolynomialFeatures(
                self.polynomial_degree, include_bias=False)))
        num_pipe = Pipeline(nsteps) if nsteps else "passthrough"

        # categorical pipeline
        csteps = []
        if self.rare_threshold:
            csteps.append(("rare", RareCategory(self.rare_threshold)))
        enc_map = {
            "onehot": OneHotEncoder(handle_unknown="ignore", sparse=False),
            "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            "target": TargetEncoder(),
            "woe": WOEEncoder(),
            "hash": HashingEncoder(),
            "freq": FrequencyEncoder(),
            "none": "passthrough"}
        csteps.append(("enc", enc_map[self.cat_encoding]))
        cat_pipe = Pipeline(csteps)

        # text pipeline
        text_map = {"tfidf": TfidfVectorizer,
                    "count": CountVectorizer, "hashing": HashingVectorizer}
        transformers = [("num", num_pipe, num),
                        ("cat", cat_pipe, cat)]
        for tcol in txt:
            transformers.append(
                (f"text_{tcol}", text_map[self.text_vectorizer](**self.text_params), tcol))
        # datetime
        if dt:
            transformers.append(("dt", DatetimeExpand(drop=False), dt))
        # cyclical
        for col, period in cyc.items():
            transformers.append((f"cyc_{col}", Cyclical(period), [col]))
        # date delta
        for col, ref in delta.items():
            transformers.append((f"delta_{col}", DateDelta(ref), [col]))

        pre = ColumnTransformer(transformers, remainder="drop")
        self.pipe_ = Pipeline([("pre", pre)])

        # attach post-aggregation or custom if present
        for func in self.custom_steps:
            self.pipe_.steps.append((func.__name__, FunctionTransformer(
                lambda X_df, func=func: func(pd.DataFrame(X_df)), feature_names_out="one-to-one")))

    # ---------------- public api ---------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if self.target and self.target in X.columns:
            y = y or X[self.target]
            X = X.drop(columns=[self.target])
        self._build(X)
        self.pipe_.fit(X, y)
        return self

    def transform(self, X): return self.pipe_.transform(X)
    def fit_transform(self, X, y=None): return self.fit(X, y).transform(X)

    def save(self):
        self.save_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.pipe_, self.save_path)
