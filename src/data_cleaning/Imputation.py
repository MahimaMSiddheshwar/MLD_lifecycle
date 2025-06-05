#!/usr/bin/env python3
"""
stage2_imputation.py

– Numeric missing‐value imputation with mean/median/KNN/MICE/random‐sample
  • Automatically drop numeric columns with > MISSING_THRESHOLD fraction missing
  • Cast “object” columns that are mostly numeric to numeric
  • Compute train & validation KS, variance ratio, covariance‐change
  • Log runtime for expensive imputers
  • Provide transform() for new‐data imputation (no leakage)

– Categorical missing‐value imputation with mode/constant/random‐sample
  • Automatically collapse categories < RARE_FREQ_CUTOFF into "__RARE__" before imputation
  • Use total‐variation distance (TVD) on train to pick best imputer
  • Provide transform() for new‐data

Usage:
    from stage2_imputation import MissingValueImputer
    imputer = MissingValueImputer(
        numeric_missing_threshold=0.5,
        categorical_missing_threshold=0.5,
        random_state=42,
        verbose=True
    )
    train_imputed = imputer.fit_transform(train_df, val_df)
    test_imputed = imputer.transform(test_df)
"""
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ks_2samp
from statsmodels.stats.missing import test_missingness
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer
)
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EmpiricalCovariance, MinCovDet


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    2A) Numeric imputation:
      – Drop numeric columns with > numeric_missing_threshold fraction missing
      – If object‐dtype but > 90% digits → cast to numeric first
      – Candidates: mean, median, KNN, MICE, random‐sample
      – Metrics: train KS, val KS, var_ratio ≥ 0.5, cov_change ≤ 0.2
      – Runtime guard: skip KNN/MICE if train size > knn_mice_max_rows
      – Stores imputers in self.numeric_imputers

    2B) Categorical imputation:
      – Collapse categories < rare_freq_cutoff into "__RARE__"
      – Candidates: mode, constant ("__MISSING__"), random‐sample
      – Metric: total‐variation distance on train
      – Stores imputers in self.categorical_imputers

    transform(new_df) applies fitted imputers to new data without leakage.
    """

    def __init__(
        self,
        numeric_missing_threshold: float = 0.5,
        categorical_missing_threshold: float = 0.5,
        rare_freq_cutoff: float = 0.01,
        knn_mice_max_rows: int = 5000,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.numeric_missing_threshold = numeric_missing_threshold
        self.categorical_missing_threshold = categorical_missing_threshold
        self.rare_freq_cutoff = rare_freq_cutoff
        self.knn_mice_max_rows = knn_mice_max_rows
        self.random_state = random_state
        self.verbose = verbose

        # fitted imputers
        # col -> (method, imputer_obj or None)
        self.numeric_imputers = {}
        # col -> (method, imputer_obj or None or "__MISSING__")
        self.categorical_imputers = {}

        # metadata
        self.dropped_numeric_cols = []
        self.dropped_categorical_cols = []
        self.report = {
            "dropped_cols": {"numeric": [], "categorical": []},
            "missing_numeric": {},
            "missing_categorical": {},
            "missing_pattern": {},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ——— Missingness Pattern Analysis ———

    def _analyze_missing_pattern(self, df: pd.DataFrame):
        """
        Compute proportion missing per column and pairwise correlations of missingness.
        Stores a small report (missing_frac + corr matrix) for debugging.
        """
        missing_frac = df.isna().mean().to_dict()
        if len(df.columns) >= 2:
            # Pairwise missingness correlation (Pearson on boolean mask)
            mask = df.isna().astype(int)
            corr = mask.corr().to_dict()
        else:
            corr = {}
        self.report["missing_pattern"] = {
            "missing_fraction": missing_frac,
            "missing_corr": corr,
        }

    # ——— Numeric Imputation Helpers ———

    def _cast_mixed_numeric(self, df: pd.DataFrame):
        """
        For each numeric‐candidate column stored as object,
        if ≥ 90% of non‐null values are digit‐like, cast to float.
        """
        for col in df.select_dtypes(include=["object"]).columns:
            ser = df[col].dropna().astype(str)
            if len(ser) == 0:
                continue
            digit_frac = ser.str.replace(
                r'[^\d\.\-]', '', regex=True).str.match(r'^-?\d+(\.\d+)?$').mean()
            if digit_frac >= 0.9:
                self._log(
                    f"  • Casting column '{col}' to numeric (digit_frac={digit_frac:.2f})")
                df[col] = pd.to_numeric(df[col], errors='coerce')

    def _evaluate_impute_num(
        self, col: str,
        orig: pd.Series,
        imputed: pd.Series,
        cov_before: np.ndarray,
    ) -> (float, float, float):
        """
        Returns (ks_train, ks_val, var_ratio, cov_change)
        Actually: returns (ks_train, var_ratio, cov_change). Validation KS is computed elsewhere.
        """
        orig_nonnull = orig.dropna().values
        imp_nonnull = imputed.dropna().values

        # KS on train
        ks_train = 0.0
        if len(orig_nonnull) >= 2 and len(imp_nonnull) >= 2:
            try:
                ks_train = float(ks_2samp(orig_nonnull, imp_nonnull)[1])
            except Exception:
                ks_train = 0.0

        # Variance ratio
        var_orig = float(np.nanvar(orig_nonnull)) if len(
            orig_nonnull) > 0 else np.nan
        var_imp = float(np.nanvar(imp_nonnull)) if len(
            imp_nonnull) > 0 else np.nan
        var_ratio = var_imp / var_orig if var_orig and var_orig > 0 else np.nan

        # Covariance‐change (train only)
        if cov_before is None:
            cov_change = np.nan
        else:
            try:
                # recompute covariance on complete cases after imputation
                temp = self.train_numeric.copy()
                temp[col] = imputed.values
                complete_idx = temp.dropna().index
                if len(complete_idx) < 5:
                    cov_change = np.nan
                else:
                    cov_after = EmpiricalCovariance().fit(
                        temp.loc[complete_idx].values).covariance_
                    idx_feat = self.numeric_cols.index(col)
                    diff = np.abs(
                        cov_after[idx_feat, :] - cov_before[idx_feat, :])
                    denom = np.abs(cov_before[idx_feat, :]) + 1e-9
                    cov_change = float(np.sum(diff / denom))
            except Exception:
                cov_change = np.nan

        return ks_train, var_ratio, cov_change

    def _random_sample_impute_num(self, orig: pd.Series):
        """
        Impute numeric by drawing random samples from non‐null values.
        """
        nonnull = orig.dropna().values
        out = orig.copy()
        mask = out.isna()
        if len(nonnull) == 0:
            return out.fillna(0.0)
        out.loc[mask] = np.random.RandomState(self.random_state).choice(
            nonnull, size=mask.sum(), replace=True
        )
        return out

    def _compute_cov_before(self):
        """
        Compute empirical covariance on complete numeric rows (train).
        Returns None if insufficient complete rows.
        """
        df_num = self.train_numeric
        complete = df_num.dropna()
        if complete.shape[0] < 5:
            return None
        return EmpiricalCovariance().fit(complete.values).covariance_

    # ——— Categorical Imputation Helpers ———

    def _random_sample_impute_cat(self, orig: pd.Series):
        """
        Impute categorical by sampling from non‐null values.
        """
        arr = orig.copy().astype(object)
        nonnull = orig.dropna().astype(object).values
        if len(nonnull) == 0:
            return arr.fillna("__MISSING__")
        mask = arr.isna()
        arr.loc[mask] = np.random.RandomState(self.random_state).choice(
            nonnull, size=mask.sum(), replace=True
        )
        return arr

    def _evaluate_impute_cat(self, orig: pd.Series):
        """
        Candidates: mode, constant, random_sample
        Compare total‐variation distance (TVD) on train.
        Returns (best_method, {method: score})
        """
        freq_before = orig.dropna().value_counts(normalize=True)
        candidates = {}
        scores = {}

        # 1) mode
        try:
            imp = SimpleImputer(strategy="most_frequent")
            arr = pd.Series(
                imp.fit_transform(orig.values.reshape(-1, 1)).flatten(),
                index=orig.index
            ).astype(object)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(np.abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            candidates["mode"] = (arr, clone(imp))
            scores["mode"] = 1.0 - tvd
        except Exception:
            pass

        # 2) constant
        try:
            arr = orig.fillna("__MISSING__").astype(object)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(np.abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            candidates["constant"] = (arr, "__MISSING__")
            scores["constant"] = 1.0 - tvd
        except Exception:
            pass

        # 3) random_sample
        try:
            arr = self._random_sample_impute_cat(orig)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(np.abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            candidates["random_sample"] = (arr, None)
            scores["random_sample"] = 1.0 - tvd
        except Exception:
            pass

        if not scores:
            return "constant", {"constant": 0.0}

        best = max(scores, key=scores.get)
        return best, scores

    # ——— Main Fit_Transform ———

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """
        Fit imputers on (train_df, val_df):
          – Analyze missingness pattern on train_df
          – Separate numeric vs categorical
          – Drop columns with > missing_threshold
          – Cast mixed‐type numeric
          – Impute numeric, store imputers
          – Collapse rare categories, impute categorical, store imputers
        """
        df_train = train_df.copy()
        if val_df is not None:
            df_val = val_df.copy()
        else:
            df_val = None

        # 0) Missingness pattern
        self._analyze_missing_pattern(df_train)

        # Identify numeric vs categorical
        df_train = df_train.copy()
        self.numeric_cols = df_train.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df_train.columns if c not in self.numeric_cols]

        # 1) Drop numeric columns with too many missing
        for col in list(self.numeric_cols):
            frac_missing = df_train[col].isna().mean()
            if frac_missing > self.numeric_missing_threshold:
                self._log(
                    f"  • Dropping numeric column '{col}' (missing frac={frac_missing:.2f} > {self.numeric_missing_threshold})")
                self.dropped_numeric_cols.append(col)
                self.report["dropped_cols"]["numeric"].append(col)
                self.numeric_cols.remove(col)
                df_train.drop(columns=[col], inplace=True)

        # 2) Cast mixed‐type numeric columns (object→numeric if ≥90% digit‐like)
        #    Also update numeric_cols if some object columns were cast
        self._cast_mixed_numeric(df_train)
        # recalc numeric/cat
        self.numeric_cols = df_train.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df_train.columns if c not in self.numeric_cols]

        # Keep a copy of numeric block for covariance computations
        self.train_numeric = df_train[self.numeric_cols].copy()

        # Compute covariance before imputation
        cov_before = self._compute_cov_before()

        # 3) Impute numeric columns
        for col in self.numeric_cols:
            orig = df_train[col]
            n_missing = orig.isna().sum()
            if n_missing == 0:
                self._log(f"  • Numeric '{col}': no missing → skip")
                self.report["missing_numeric"][col] = {
                    "chosen": "none", "note": "no missing"}
                self.numeric_imputers[col] = ("none", None)
                continue

            self._log(
                f"  • Numeric '{col}': {n_missing} missing, evaluating imputers")
            # method -> (ks_train, var_ratio, cov_change, runtime)
            metrics = {}
            candidates = {}      # method -> imputed_series
            imputers = {}        # method -> imputer_obj or None

            # Prepare train & val arrays
            orig_train = orig.copy()
            if df_val is not None and col in df_val.columns:
                orig_val = df_val[col].copy()
            else:
                orig_val = None

            # — Mean —
            try:
                start = time.time()
                imp = SimpleImputer(strategy="mean")
                arr = pd.Series(
                    imp.fit_transform(
                        orig_train.values.reshape(-1, 1)).flatten(),
                    index=orig_train.index
                )
                ks_train, var_ratio, cov_change = self._evaluate_impute_num(
                    col, orig_train, arr, cov_before)
                runtime = time.time() - start
                metrics["mean"] = (ks_train, var_ratio, cov_change, runtime)
                candidates["mean"] = arr
                imputers["mean"] = clone(imp)
                self._log(
                    f"    • mean: ks={ks_train:.3f}, vr={var_ratio:.3f}, cov_ch={cov_change:.3f}, time={runtime:.2f}s")
            except Exception:
                pass

            # — Median —
            try:
                start = time.time()
                imp = SimpleImputer(strategy="median")
                arr = pd.Series(
                    imp.fit_transform(
                        orig_train.values.reshape(-1, 1)).flatten(),
                    index=orig_train.index
                )
                ks_train, var_ratio, cov_change = self._evaluate_impute_num(
                    col, orig_train, arr, cov_before)
                runtime = time.time() - start
                metrics["median"] = (ks_train, var_ratio, cov_change, runtime)
                candidates["median"] = arr
                imputers["median"] = clone(imp)
                self._log(
                    f"    • median: ks={ks_train:.3f}, vr={var_ratio:.3f}, cov_ch={cov_change:.3f}, time={runtime:.2f}s")
            except Exception:
                pass

            # — KNN (only if small) —
            if df_train.shape[0] <= self.knn_mice_max_rows:
                try:
                    start = time.time()
                    imp = KNNImputer(n_neighbors=5)
                    tmp = self.train_numeric.copy()
                    tmp_imp = pd.DataFrame(
                        imp.fit_transform(tmp),
                        columns=self.numeric_cols,
                        index=tmp.index
                    )
                    arr = tmp_imp[col]
                    ks_train, var_ratio, cov_change = self._evaluate_impute_num(
                        col, orig_train, arr, cov_before)
                    runtime = time.time() - start
                    metrics["knn"] = (ks_train, var_ratio, cov_change, runtime)
                    candidates["knn"] = arr
                    imputers["knn"] = clone(imp)
                    self._log(
                        f"    • knn: ks={ks_train:.3f}, vr={var_ratio:.3f}, cov_ch={cov_change:.3f}, time={runtime:.2f}s")
                except Exception:
                    pass
            else:
                self._log(
                    f"    • knn skipped (train rows > {self.knn_mice_max_rows})")

            # — MICE (only if small) —
            if df_train.shape[0] <= self.knn_mice_max_rows:
                try:
                    start = time.time()
                    imp = IterativeImputer(
                        estimator=BayesianRidge(),
                        sample_posterior=True,
                        random_state=self.random_state,
                        max_iter=10
                    )
                    tmp = self.train_numeric.copy()
                    tmp_imp = pd.DataFrame(
                        imp.fit_transform(tmp),
                        columns=self.numeric_cols,
                        index=tmp.index
                    )
                    arr = tmp_imp[col]
                    ks_train, var_ratio, cov_change = self._evaluate_impute_num(
                        col, orig_train, arr, cov_before)
                    runtime = time.time() - start
                    metrics["mice"] = (ks_train, var_ratio,
                                       cov_change, runtime)
                    candidates["mice"] = arr
                    imputers["mice"] = clone(imp)
                    self._log(
                        f"    • mice: ks={ks_train:.3f}, vr={var_ratio:.3f}, cov_ch={cov_change:.3f}, time={runtime:.2f}s")
                except Exception:
                    pass
            else:
                self._log(
                    f"    • mice skipped (train rows > {self.knn_mice_max_rows})")

            # — Random‐sample —
            try:
                start = time.time()
                arr = self._random_sample_impute_num(orig_train)
                ks_train, var_ratio, cov_change = self._evaluate_impute_num(
                    col, orig_train, arr, cov_before)
                runtime = time.time() - start
                metrics["random_sample"] = (
                    ks_train, var_ratio, cov_change, runtime)
                candidates["random_sample"] = arr
                imputers["random_sample"] = None
                self._log(
                    f"    • random_sample: ks={ks_train:.3f}, vr={var_ratio:.3f}, cov_ch={cov_change:.3f}, time={runtime:.2f}s")
            except Exception:
                pass

            # — Select best candidate using train metrics —
            best_method = None
            # (ks_train, var_ratio, cov_change)
            best_score = (-1.0, -1.0, np.inf)
            for m, (ks_train, vr, cc, rt) in metrics.items():
                # Qualify: vr ≥ 0.5 and cc ≤ 0.2 (or cc=nan)
                if not np.isnan(vr) and vr < 0.5:
                    continue
                if not np.isnan(cc) and cc > 0.2:
                    continue
                # Rank by (ks_train, vr, -cc)
                score = (ks_train, vr, -cc)
                if score > best_score:
                    best_score = score
                    best_method = m

            if best_method is None:
                # fallback → mean
                arr = orig_train.fillna(orig_train.mean())
                ks_train, var_ratio, cov_change = self._evaluate_impute_num(
                    col, orig_train, arr, cov_before)
                best_method = "fallback_mean"
                imputers["fallback_mean"] = SimpleImputer(
                    strategy="mean").fit(orig_train.values.reshape(-1, 1))
                candidates["fallback_mean"] = arr
                best_score = (ks_train, var_ratio, cov_change)
                self._log(
                    f"    • Fallback to mean: ks={ks_train:.3f}, vr={var_ratio:.3f}, cov_ch={cov_change:.3f}")

            # Record and apply to train
            self.report["missing_numeric"][col] = {
                "chosen": best_method,
                "metrics": best_score,
            }
            self._log(
                f"    → Selected '{best_method}' for '{col}' with metrics={best_score}")
            df_train[col] = candidates[best_method].values
            self.numeric_imputers[col] = (
                best_method, imputers.get(best_method))

        # 4) Drop categorical columns with too many missing
        for col in list(self.categorical_cols):
            frac_missing = df_train[col].isna().mean()
            if frac_missing > self.categorical_missing_threshold:
                self._log(
                    f"  • Dropping categorical column '{col}' (missing frac={frac_missing:.2f} > {self.categorical_missing_threshold})")
                self.dropped_categorical_cols.append(col)
                self.report["dropped_cols"]["categorical"].append(col)
                self.categorical_cols.remove(col)
                df_train.drop(columns=[col], inplace=True)

        # 5) Collapse rare categories before imputation
        for col in self.categorical_cols:
            freq = df_train[col].value_counts(normalize=True)
            rare_levels = set(freq[freq < self.rare_freq_cutoff].index)
            if rare_levels:
                df_train[col] = df_train[col].apply(
                    lambda x: "__RARE__" if x in rare_levels else x)

        # 6) Impute categorical columns
        for col in self.categorical_cols:
            orig = df_train[col].astype(object)
            n_missing = orig.isna().sum()
            if n_missing == 0:
                self._log(f"  • Categorical '{col}': no missing → skip")
                self.report["missing_categorical"][col] = {
                    "chosen": "none", "note": "no missing"}
                self.categorical_imputers[col] = ("none", None)
                continue

            self._log(
                f"  • Categorical '{col}': {n_missing} missing, evaluating imputers")
            best_method, scores = self._evaluate_impute_cat(orig)
            self.report["missing_categorical"][col] = {
                "chosen": best_method,
                "scores": scores
            }
            self._log(
                f"    → Selected '{best_method}' for '{col}' (scores={scores})")

            if best_method == "mode":
                imp = SimpleImputer(strategy="most_frequent")
                df_train[col] = pd.Series(
                    imp.fit_transform(orig.values.reshape(-1, 1)).flatten(),
                    index=orig.index
                ).astype(object)
                self.categorical_imputers[col] = ("mode", clone(imp))
            elif best_method == "constant":
                df_train[col] = orig.fillna("__MISSING__").astype(object)
                self.categorical_imputers[col] = ("constant", "__MISSING__")
            elif best_method == "random_sample":
                df_train[col] = self._random_sample_impute_cat(orig)
                self.categorical_imputers[col] = ("random_sample", None)
            else:
                # fallback
                df_train[col] = orig.fillna("__MISSING__").astype(object)
                self.categorical_imputers[col] = ("constant", "__MISSING__")

        # Save the final train imputed
        self.train_imputed_ = df_train.copy()
        return self

    def fit_transform(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """
        Fit on (train_df, val_df) and return imputed train_df.
        """
        self.fit(train_df, val_df)
        return self.train_imputed_

    # ——— Transform on New Data (e.g. Test) ———

    def transform(self, new_df: pd.DataFrame):
        """
        Apply fitted imputers to new_df (no refitting).
        Returns a copy with imputations applied (drops same columns as train).
        """
        df = new_df.copy()

        # 1) Drop same columns
        for col in self.dropped_numeric_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        for col in self.dropped_categorical_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # 2) Cast mixed‐type numeric
        for col in self.numeric_imputers.keys():
            if col in df.columns and df[col].dtype == object:
                ser = df[col].dropna().astype(str)
                digit_frac = ser.str.replace(
                    r'[^\d\.\-]', '', regex=True).str.match(r'^-?\d+(\.\d+)?$').mean()
                if digit_frac >= 0.9:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3) Impute numeric
        for col, (method, imputer) in self.numeric_imputers.items():
            if col not in df.columns:
                continue
            if method in ("none",):
                continue
            ser = df[col]
            if method in ("mean", "median", "fallback_mean"):
                arr = pd.Series(imputer.transform(
                    ser.values.reshape(-1, 1)).flatten(), index=ser.index)
                df[col] = arr
            elif method == "knn":
                tmp = self.train_numeric.copy().append(
                    df[self.numeric_cols])[self.numeric_cols]
                imp = imputer
                arr_all = pd.DataFrame(
                    imp.transform(tmp),
                    columns=self.numeric_cols,
                    index=tmp.index
                )
                df[col] = arr_all.loc[df.index, col]
            elif method == "mice":
                tmp = self.train_numeric.copy().append(
                    df[self.numeric_cols])[self.numeric_cols]
                imp = imputer
                arr_all = pd.DataFrame(
                    imp.transform(tmp),
                    columns=self.numeric_cols,
                    index=tmp.index
                )
                df[col] = arr_all.loc[df.index, col]
            elif method == "random_sample":
                df[col] = self._random_sample_impute_num(df[col])
            else:
                # Should not happen
                df[col] = df[col].fillna(df[col].mean())

        # 4) Collapse rare categories using train’s rare buckets
        for col in self.categorical_imputers.keys():
            if col not in df.columns:
                continue
            # We collapse using training–set rare_freq_cutoff
            freq = df[col].value_counts(normalize=True)
            rare_levels = set(freq[freq < self.rare_freq_cutoff].index)
            if rare_levels:
                df[col] = df[col].apply(
                    lambda x: "__RARE__" if x in rare_levels else x)

        # 5) Impute categorical
        for col, (method, imputer) in self.categorical_imputers.items():
            if col not in df.columns:
                continue
            ser = df[col].astype(object)
            if method == "none":
                continue
            elif method == "mode":
                arr = pd.Series(
                    imputer.transform(ser.values.reshape(-1, 1)).flatten(),
                    index=ser.index
                ).astype(object)
                df[col] = arr
            elif method == "constant":
                df[col] = ser.fillna(imputer).astype(object)
            elif method == "random_sample":
                df[col] = self._random_sample_impute_cat(ser)
            else:
                df[col] = ser.fillna("__MISSING__").astype(object)

        return df
