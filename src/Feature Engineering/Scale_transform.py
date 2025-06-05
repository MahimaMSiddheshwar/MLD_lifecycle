#!/usr/bin/env python3
"""
stage4_scaling_transformation.py

– Choose between StandardScaler, RobustScaler, MinMaxScaler based on skew/kurtosis
– For each numeric:
    • If scaled data is not “approx normal” (Shapiro p < 0.05 or |skew| > 0.75),
      evaluate transforms: none, Box‐Cox, Yeo‐Johnson, Quantile→Normal
    • Pick best by (Shapiro p, −|skew|) and apply
– Automatically drop near‐zero‐variance features (< NZV_THRESHOLD unique values)
– Skip QuantileTransformer on very large datasets (> QT_MAX_ROWS) to avoid
  runtime blowup
– Stores fitted scaler and each column’s transform for use on new data

Usage:
    from stage4_scaling_transformation import NumericTransformer
    trans = NumericTransformer(
        nzv_threshold=2,
        qt_max_rows=100_000,
        verbose=True
    )
    train_scaled = trans.fit_transform(train_df, numeric_cols)
    val_scaled = trans.transform(val_df)
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin, clone


class NumericTransformer(BaseEstimator, TransformerMixin):
    """
    4A) Numeric scaling & conditional transform:
      – Drop near‐zero‐variance columns (nunique < nzv_threshold)
      – Choose scaler based on skew/kurtosis thresholds
      – Apply scaler to train, store as self.scaler
      – For each numeric col: if scaled data fails (Shapiro p < 0.05 or |skew| > 0.75),
        evaluate “none”, Box‐Cox, Yeo‐Johnson, Quantile→Normal (subsample if > QT_MAX_ROWS)
      – Pick best by (pval, −|skew|), apply and store in self.transform_models
      – transform(new_df) applies same scaler + transforms; drops same NZV cols
    """

    def __init__(
        self,
        nzv_threshold: int = 2,
        sk_thresh_robust: float = 1.0,
        kurt_thresh_robust: float = 5.0,
        sk_thresh_standard: float = 0.5,
        transform_skew_thresh: float = 0.75,
        shapiro_p_thresh: float = 0.05,
        qt_max_rows: int = 100_000,
        verbose: bool = False,
        random_state: int = 42,
    ):
        self.nzv_threshold = nzv_threshold
        self.sk_thresh_robust = sk_thresh_robust
        self.kurt_thresh_robust = kurt_thresh_robust
        self.sk_thresh_standard = sk_thresh_standard
        self.transform_skew_thresh = transform_skew_thresh
        self.shapiro_p_thresh = shapiro_p_thresh
        self.qt_max_rows = qt_max_rows
        self.verbose = verbose
        self.random_state = random_state

        # Will be set in fit
        self.numeric_cols = []
        self.scaler = None
        # col -> ("none"/"boxcox"/"yeo"/"quantile", fitted_obj or None)
        self.transform_models = {}
        self.nzv_cols_ = []
        self.report = {
            "nzv_cols": [],
            "chosen_scaler": None,
            "scaler_stats": {},
            "transforms": {},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ——— Helper to choose scaler ———

    def _choose_scaler(self, df: pd.DataFrame):
        """
        Decide among StandardScaler, RobustScaler, MinMaxScaler.
        Returns (scaler_name, stats_dict).
        """
        skews = {}
        kurts = {}
        for col in self.numeric_cols:
            arr = df[col].dropna().values
            if len(arr) > 2:
                skews[col] = float(skew(arr))
                kurts[col] = float(kurtosis(arr))
            else:
                skews[col] = 0.0
                kurts[col] = 0.0

        stats_dict = {"skew": skews, "kurtosis": kurts}

        # If any column is heavy‐tailed → RobustScaler
        for col in self.numeric_cols:
            if abs(skews[col]) > self.sk_thresh_robust or abs(kurts[col]) > self.kurt_thresh_robust:
                return "RobustScaler", stats_dict

        # If all columns are nearly normal → StandardScaler
        if all(abs(sk) < self.sk_thresh_standard for sk in skews.values()):
            return "StandardScaler", stats_dict

        # Otherwise → MinMaxScaler
        return "MinMaxScaler", stats_dict

    # ——— Evaluate transforms for one column ———

    def _evaluate_transform(self, arr: np.ndarray):
        """
        On a 1D array arr, test candidates: none, boxcox, yeo, quantile.
        Returns (best_method, {method: (pval, abs(skew))}).
        Subsample if arr size > qt_max_rows for QuantileTransformer.
        """
        scores = {}
        N = len(arr)
        if N < 5 or np.unique(arr).size < 5:
            return "none", {"none": (1.0, 0.0)}

        # Precompute sample for Shapiro
        if N > self.qt_max_rows:
            rng = np.random.RandomState(self.random_state)
            subsample = rng.choice(arr, size=self.qt_max_rows, replace=False)
        else:
            subsample = arr

        for method in ["none", "boxcox", "yeo", "quantile"]:
            try:
                if method == "none":
                    arr_t = arr.copy()
                elif method == "boxcox":
                    if np.any(arr <= 0):
                        continue
                    arr_t = stats.boxcox(arr)[0]
                elif method == "yeo":
                    pt = PowerTransformer(
                        method="yeo-johnson", standardize=True)
                    arr_t = pt.fit_transform(arr.reshape(-1, 1)).flatten()
                else:  # quantile
                    if N > self.qt_max_rows:
                        qt = QuantileTransformer(
                            output_distribution="normal",
                            random_state=self.random_state,
                            subsample=self.qt_max_rows
                        )
                    else:
                        qt = QuantileTransformer(
                            output_distribution="normal",
                            random_state=self.random_state
                        )
                    arr_t = qt.fit_transform(arr.reshape(-1, 1)).flatten()

                # Shapiro on subsample
                if subsample.size >= 3:
                    stat, pval = stats.shapiro(
                        subsample if method == "none" else arr_t[:subsample.size])
                else:
                    pval = 0.0
                skew_abs = abs(float(skew(arr_t)))
                scores[method] = (pval, skew_abs)
            except Exception:
                continue

        # Select best by (pval, −skew_abs)
        best, best_score = "none", (-1.0, np.inf)
        for m, (pval, sk_abs) in scores.items():
            score = (pval, -sk_abs)
            if score > best_score:
                best, best_score = m, score

        return best, scores

    # ——— Main fit_transform ———

    def fit(self, df: pd.DataFrame, numeric_cols: list):
        """
        Fit on training data:
         – Drop columns with < nzv_threshold unique non‐null values
         – Choose & fit scalar, then transform
         – For each column: decide if extra transform is needed; fit & apply it
        """
        df_num = df[numeric_cols].copy()

        # 0) Drop NZV columns
        nzv = [col for col in numeric_cols if df_num[col].nunique(
            dropna=True) < self.nzv_threshold]
        self.nzv_cols_ = nzv
        self.report["nzv_cols"] = nzv
        self._log(f"  • Dropping NZV columns: {nzv}")
        self.numeric_cols = [c for c in numeric_cols if c not in nzv]
        df_num.drop(columns=nzv, inplace=True)

        if not self.numeric_cols:
            self.report["chosen_scaler"] = "none"
            self.report["scaler_stats"] = {}
            self.report["transforms"] = {}
            self.train_transformed_ = df.copy()
            return self

        # 1) Choose scaler
        scaler_name, stats_dict = self._choose_scaler(df_num)
        self.report["chosen_scaler"] = scaler_name
        self.report["scaler_stats"] = stats_dict
        self._log(f"  • Chosen scaler: {scaler_name}")

        if scaler_name == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_name == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        # 2) Fit & transform scaler
        X_scaled = scaler.fit_transform(df_num.values)
        self.scaler = scaler
        df_scaled = pd.DataFrame(
            X_scaled, columns=self.numeric_cols, index=df.index)

        # 3) Conditional transforms per column
        self.report["transforms"] = {}
        df_out = df.copy()
        df_out[self.numeric_cols] = df_scaled

        for col in self.numeric_cols:
            arr = df_scaled[col].dropna().values
            if arr.size < 5:
                self.report["transforms"][col] = {
                    "chosen": "none", "scores": {"none": (1.0, 0.0)}}
                self._log(f"  • '{col}': too few non‐null to transform")
                continue

            # Check if scaled is already approx normal
            try:
                if arr.size > self.qt_max_rows:
                    rng = np.random.RandomState(self.random_state)
                    sample = rng.choice(
                        arr, size=self.qt_max_rows, replace=False)
                else:
                    sample = arr
                pval = stats.shapiro(sample)[1] if sample.size >= 3 else 0.0
            except Exception:
                pval = 0.0
            sk_abs = abs(float(skew(arr)))

            if pval > self.shapiro_p_thresh and sk_abs < self.transform_skew_thresh:
                # no transform needed
                self.report["transforms"][col] = {
                    "chosen": "none", "scores": {"none": (pval, sk_abs)}}
                self._log(
                    f"  • '{col}': no extra transform (p={pval:.3f}, skew={sk_abs:.3f})")
                self.transform_models[col] = ("none", None)
                continue

            # Evaluate candidates
            best, scores = self._evaluate_transform(arr)
            self.report["transforms"][col] = {"chosen": best, "scores": scores}
            self._log(f"  • '{col}': best transform={best}, scores={scores}")

            if best == "boxcox":
                if np.all(arr > 0):
                    arr_t = stats.boxcox(arr)[0]
                    df_out[col] = arr_t
                    self.transform_models[col] = ("boxcox", None)
                else:
                    self._log(
                        f"    • '{col}': boxcox infeasible (non‐positive values)")
                    self.transform_models[col] = ("none", None)
            elif best == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                arr_t = pt.fit_transform(arr.reshape(-1, 1)).flatten()
                df_out[col] = arr_t
                self.transform_models[col] = ("yeo", clone(pt))
            elif best == "quantile":
                if arr.size > self.qt_max_rows:
                    qt = QuantileTransformer(
                        output_distribution="normal",
                        random_state=self.random_state,
                        subsample=self.qt_max_rows
                    )
                else:
                    qt = QuantileTransformer(
                        output_distribution="normal",
                        random_state=self.random_state
                    )
                arr_t = qt.fit_transform(arr.reshape(-1, 1)).flatten()
                df_out[col] = arr_t
                self.transform_models[col] = ("quantile", clone(qt))
            else:
                # no transform
                self.transform_models[col] = ("none", None)

        self.train_transformed_ = df_out.copy()
        return self

    def fit_transform(self, df: pd.DataFrame, numeric_cols: list):
        return self.fit(df, numeric_cols).train_transformed_

    # ——— Transform New Data ———

    def transform(self, df: pd.DataFrame):
        """
        Apply the same scaler and transforms to new df.
        Drops NZV columns; applies scaler; applies each col’s transform.
        """
        df_out = df.copy()

        # Drop NZV columns
        for col in self.nzv_cols_:
            if col in df_out.columns:
                df_out.drop(columns=[col], inplace=True)

        if not self.numeric_cols:
            return df_out

        df_num = df_out[self.numeric_cols].copy()

        # 1) Scale
        X_scaled = self.scaler.transform(df_num.values)
        df_scaled = pd.DataFrame(
            X_scaled, columns=self.numeric_cols, index=df_out.index)
        df_out[self.numeric_cols] = df_scaled

        # 2) Apply transforms
        for col, (method, model) in self.transform_models.items():
            if col not in df_out.columns:
                continue
            arr = df_out[col].values
            if method == "boxcox":
                if np.all(arr > 0):
                    arr_t = stats.boxcox(arr)[0]
                    df_out[col] = arr_t
                else:
                    self._log(
                        f"    • '{col}': boxcox infeasible on new data (non‐positive values)")
            elif method == "yeo":
                pt = model
                df_out[col] = pt.transform(arr.reshape(-1, 1)).flatten()
            elif method == "quantile":
                qt = model
                df_out[col] = qt.transform(arr.reshape(-1, 1)).flatten()
            # else “none”: do nothing

        return df_out
