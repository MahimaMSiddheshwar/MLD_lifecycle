#!/usr/bin/env python3
"""
stage3_outlier_detection.py

– Univariate outliers: IQR, Z-score, Modified Z, Tukey (2×IQR), 1st/99th percentile
– Multivariate outliers: Mahalanobis on scaled data (with MinCovDet fallback),
  LocalOutlierFactor, and IsolationForest as last resort
– Vote-based: if a row is flagged by ≥ outlier_threshold methods, it is a “real outlier”
– fit/fit_transform: detect + (drop or winsorize) automatically, with full reporting
– transform(): flag outliers (no removal) on new data
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detect & treat outliers in one step (fit/fit_transform), then flag on new data via transform().

    – Univariate fences: IQR, Z-score, Modified Z, Tukey (2× IQR), 1st/99th percentile
    – Multivariate fences (Mahalanobis → MinCovDet fallback, LOF, IsolationForest)
    – “Real outlier” if flagged by ≥ outlier_threshold rules
    – fit/fit_transform: automatically drop or winsorize rows (unless cap_outliers=None)
    – transform(): flag (no removal) on new data, producing an “is_outlier” column

    Parameters
    ----------
    outlier_threshold : int
        Minimum vote count for a row to be considered a “real outlier.”
    robust_covariance : bool
        If True, will fallback to MinCovDet if EmpiricalCovariance flags >5% of rows.
    cap_outliers : Optional[bool]
        If True, winsorize detected rows (clip to 1st/99th percentiles).
        If False, drop detected rows outright.
        If None, do not remove or cap (only detect & report).
    model_family : Optional[str]
        If "linear" or "bayesian", we force winsorization (unless cap_outliers=False is explicitly set).
        Otherwise ignored. Default None.
    random_state : int
        Seed for IsolationForest, MinCovDet, and LOF.
    verbose : bool
        If True, prints basic log messages during detection & treatment.
    """

    # ─────────────── Class-Level Constants ───────────────
    # Univariate thresholds
    UNIV_IQR_FACTOR = 1.5    # multiplier for IQR fences
    UNIV_ZSCORE_CUTOFF = 3.0    # |z| > 3.0
    UNIV_MODZ_CUTOFF = 3.5    # |modified_z| > 3.5
    TUKEY_MULTIPLIER = 2.0    # Tukey uses 2× IQR
    PCTL_LOW = 1      # 1st percentile
    PCTL_HIGH = 99     # 99th percentile

    # Multivariate settings
    GAUSS_SKEW_THRESH = 1.0    # abs(skew) < 1.0
    GAUSS_KURT_THRESH = 5.0    # abs(kurtosis) < 5.0
    MAHA_MIN_RATIO = 5      # need n_samples ≥ 5 * n_features for Mahalanobis
    GAUSS_FRAC_THRESH = 0.6    # need ≥ 60% of features approx Gaussian
    LOF_MAX_SAMPLES = 2000   # only run LOF if n_samples < 2000
    LOF_MAX_FEATURES = 50     # only run LOF if n_features < 50
    ISO_CONTAMINATION = 0.01   # 1% contamination for IsolationForest

    MULTI_CI = 0.975  # confidence for Mahalanobis cutoff

    def __init__(
        self,
        outlier_threshold: int = 3,
        robust_covariance: bool = True,
        cap_outliers: bool = True,
        model_family: str = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.outlier_threshold = outlier_threshold
        self.robust_covariance = robust_covariance
        self.cap_outliers = cap_outliers
        self.model_family = model_family
        self.random_state = random_state
        self.verbose = verbose

        # These fields will be set during fit():
        self.df: pd.DataFrame = None
        self.numeric_cols: List[str] = []
        self.scaler = None
        self.cov_estimator = None
        self.mahal_threshold = None
        self.lof_model = None
        self.iso_model = None

        # After fit, we store:
        self.train_clean_: pd.DataFrame = None       # post-treatment training set
        self.votes_table_: pd.DataFrame = None       # per-row rule votes
        # how many values clipped per numeric column
        self.clipped_counts_: Dict[str, int] = {}

        # Reporting
        self.report: Dict[str, Any] = {
            "univariate_outliers": {},    # {column -> {rule_name: count_flagged, ...}, ...}
            # {"method": str, "indices": [...], "fallback_path": [...]}
            "multivariate_outliers": {},
            "real_outliers": {},          # {"indices": [...], "count": N}
            "treatment": {}               # details about drop vs winsorize
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ───────────── Univariate Outlier Rules ─────────────

    def _iqr_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - self.UNIV_IQR_FACTOR * iqr, q3 + self.UNIV_IQR_FACTOR * iqr
        return series[(series < lb) | (series > ub)].index.tolist()

    def _zscore_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size < 2:
            return []
        mu, sigma = series.mean(), series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return []
        z = (series - mu) / sigma
        return series[np.abs(z) > self.UNIV_ZSCORE_CUTOFF].index.tolist()

    def _modz_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size < 2:
            return []
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0:
            return []
        modz = 0.6745 * (series - med) / mad
        return series[np.abs(modz) > self.UNIV_MODZ_CUTOFF].index.tolist()

    def _tukey_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - self.TUKEY_MULTIPLIER * iqr, q3 + self.TUKEY_MULTIPLIER * iqr
        return series[(series < lb) | (series > ub)].index.tolist()

    def _percentile_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size == 0:
            return []
        p1, p99 = np.percentile(arr, [self.PCTL_LOW, self.PCTL_HIGH])
        return series[(series < p1) | (series > p99)].index.tolist()

    # ───────────── Multivariate Outlier Routines ─────────────

    def _fit_mahalanobis(self, df_numeric: pd.DataFrame):
        """
        1) Z-score numeric block → self.scaler
        2) Fit EmpiricalCovariance on z-scored data → self.cov_estimator
        3) Compute χ² cutoff at MULTI_CI (default 97.5%) → self.mahal_threshold
        """
        scaler = StandardScaler()
        Xz = scaler.fit_transform(df_numeric.values)
        self.scaler = scaler

        try:
            emp = EmpiricalCovariance().fit(Xz)
            self.cov_estimator = emp
        except Exception:
            self.cov_estimator = None

        p = Xz.shape[1]
        self.mahal_threshold = chi2.ppf(self.MULTI_CI, df=p)

    def _compute_mahalanobis_indices(self, df_numeric: pd.DataFrame) -> List[int]:
        """
        1) Transform df_numeric via self.scaler
        2) Compute MDs via self.cov_estimator.mahalanobis(...)
        3) If >5% of rows flagged and robust_covariance=True → refit with MinCovDet
        4) Return list of row indices exceeding threshold
        """
        if self.cov_estimator is None:
            return []

        Xz = self.scaler.transform(df_numeric.values)
        md = self.cov_estimator.mahalanobis(Xz)
        flagged = list(df_numeric.index[md > self.mahal_threshold])

        frac_flagged = len(flagged) / float(df_numeric.shape[0])
        if self.robust_covariance and frac_flagged > 0.05:
            # Fallback to MinCovDet
            try:
                mcd = MinCovDet(random_state=self.random_state).fit(Xz)
                self.cov_estimator = mcd
                md2 = mcd.mahalanobis(Xz)
                flagged = list(df_numeric.index[md2 > self.mahal_threshold])
            except Exception:
                # If MinCovDet fails, keep the original flags
                pass

        return flagged

    def _fit_lof(self, df_numeric: pd.DataFrame):
        """
        Fit a LocalOutlierFactor model in novelty mode so we can call .predict(...) on new data.
        Contamination fixed at ISO_CONTAMINATION. 
        """
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.ISO_CONTAMINATION,
            novelty=True
        )
        lof.fit(df_numeric.values)
        self.lof_model = lof

    def _compute_lof_indices(self, df_numeric: pd.DataFrame) -> List[int]:
        """
        Return row indices where LOF.predict(...) == -1 (outlier).
        """
        if self.lof_model is None:
            return []
        preds = self.lof_model.predict(df_numeric.values)
        return list(df_numeric.index[preds == -1])

    def _fit_isolation_forest(self, df_numeric: pd.DataFrame):
        """
        Fit an IsolationForest on df_numeric with contamination=ISO_CONTAMINATION.
        """
        iso = IsolationForest(
            contamination=self.ISO_CONTAMINATION,
            random_state=self.random_state
        )
        iso.fit(df_numeric.values)
        self.iso_model = iso

    def _compute_isolation_indices(self, df_numeric: pd.DataFrame) -> List[int]:
        """
        Return row indices where IsolationForest.predict(...) == -1 (outlier).
        """
        if self.iso_model is None:
            return []
        preds = self.iso_model.predict(df_numeric.values)
        return list(df_numeric.index[preds == -1])

    def detect_multivariate_outliers(self) -> List[int]:
        """
        Chooses between Mahalanobis, LOF, or IsolationForest, in order:

          1) If n_features >= n_samples → skip Mahalanobis/LOF and go straight to IsolationForest.
          2) Else compute “Gaussian-like” fraction of columns (via |skew|<GAUSS_SKEW_THRESH and |kurtosis|<GAUSS_KURT_THRESH).
             If (n_samples >= MAHA_MIN_RATIO * n_features) and (frac_gaussian ≥ GAUSS_FRAC_THRESH):
               → run Mahalanobis. If Mahalanobis flags ≤5% of rows, accept those. Otherwise fall through.
          3) Else if (n_samples < LOF_MAX_SAMPLES) and (n_features < LOF_MAX_FEATURES):
               → run LOF. If LOF flags ≤5% of rows, accept those. Otherwise fall through.
          4) Otherwise run IsolationForest.  

        Returns a list of flagged row indices.
        """
        df_num = self.df[self.numeric_cols].copy().dropna(axis=0, how="any")
        n_samples, n_features = df_num.shape

        if n_samples < 3 or n_features == 0:
            # Not enough data or no numeric features → skip
            self.report["multivariate_outliers"] = {
                "method": None,
                "indices": [],
                "notes": "too few samples or no numeric cols"
            }
            return []

        # 1) If p >= n, skip Mahalanobis/LOF
        if n_features >= n_samples:
            self._log(
                f"Warning: n_features ({n_features}) ≥ n_samples ({n_samples}), skipping Mahalanobis/LOF")
            self._fit_isolation_forest(df_num)
            iso_idxs = self._compute_isolation_indices(df_num)
            self.report["multivariate_outliers"] = {
                "method": "IsolationForest",
                "indices": iso_idxs,
                "notes": "p >= n, direct to IsolationForest"
            }
            return iso_idxs

        # 2) Check “Gaussian-like” fraction
        skews = df_num.apply(lambda col: abs(col.dropna().skew()), axis=0)
        kurts = df_num.apply(lambda col: abs(col.dropna().kurtosis()), axis=0)
        gaussian_like = ((skews < self.GAUSS_SKEW_THRESH) &
                         (kurts < self.GAUSS_KURT_THRESH)).sum()
        frac_gaussian = gaussian_like / float(n_features)

        # Attempt Mahalanobis if conditions met
        mahal_used = False
        if (n_samples >= self.MAHA_MIN_RATIO * n_features) and (frac_gaussian >= self.GAUSS_FRAC_THRESH):
            self._fit_mahalanobis(df_num)
            maha_idxs = self._compute_mahalanobis_indices(df_num)
            frac_flagged = len(maha_idxs) / float(n_samples)
            if frac_flagged <= 0.05:
                # Accept Mahalanobis
                mahal_used = True
                self.report["multivariate_outliers"] = {
                    "method": "Mahalanobis",
                    "indices": maha_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "EmpiricalCovariance accepted"
                }
                return maha_idxs
            else:
                # Mark that we tried Mahalanobis but flagged too many; record and fall through
                self.report["multivariate_outliers"] = {
                    "method": "Mahalanobis",
                    "indices": maha_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "flags > 5%, will fall back"
                }

        # 3) Attempt LOF if still applicable
        lof_used = False
        if not mahal_used and (n_samples < self.LOF_MAX_SAMPLES) and (n_features < self.LOF_MAX_FEATURES):
            self._fit_lof(df_num)
            lof_idxs = self._compute_lof_indices(df_num)
            frac_flagged = len(lof_idxs) / float(n_samples)
            if frac_flagged <= 0.05:
                lof_used = True
                self.report["multivariate_outliers"] = {
                    "method": "LOF",
                    "indices": lof_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "LocalOutlierFactor accepted"
                }
                return lof_idxs
            else:
                # Log fallback
                self.report["multivariate_outliers"] = {
                    "method": "LOF",
                    "indices": lof_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "flags > 5%, will fall back"
                }

        # 4) Default → IsolationForest
        self._fit_isolation_forest(df_num)
        iso_idxs = self._compute_isolation_indices(df_num)
        self.report["multivariate_outliers"] = {
            "method": "IsolationForest",
            "indices": iso_idxs,
            "notes": "fallback to IsolationForest"
        }
        return iso_idxs

    # ────────────── Main Fit & Fit_Transform ──────────────

    def fit(self, df: pd.DataFrame, numeric_cols: List[str]):
        """
        1) Store original df + numeric_cols
        2) Run all univariate rules, build a per-row votes_table_
        3) Run detect_multivariate_outliers(), increment votes in votes_table_
        4) Mark “real_outliers” = rows whose total votes ≥ outlier_threshold
        5) Depending on cap_outliers / model_family, drop or winsorize those rows
           – If cap_outliers=None → no removal, simply report them
           – If model_family in ["linear","bayesian"] and cap_outliers is not False → force winsorize
           – If cap_outliers=True → winsorize
           – If cap_outliers=False → drop
        6) Save cleaned DataFrame as train_clean_
        7) Build full report, including how many clipped in winsorization
        """
        # 1) Store
        self.df = df.copy()
        self.numeric_cols = numeric_cols.copy()

        # Initialize votes_table_ with zeros
        votes_df = pd.DataFrame(0, index=self.df.index,
                                columns=[
                                    "iqr", "zscore", "modz", "tukey", "percentile",
                                    "mahalanobis", "lof", "isolation"
                                ], dtype=int)

        # Prepare report structure for univariate counts
        self.report["univariate_outliers"] = {
            col: {} for col in self.numeric_cols}

        # 2) UNIVARIATE VOTING
        for col in self.numeric_cols:
            series = self.df[col]
            col_counts = {}

            # IQR
            try:
                idxs = self._iqr_outliers(series)
                col_counts["iqr"] = len(idxs)
                for i in idxs:
                    votes_df.at[i, "iqr"] = 1
                self._log(f"  • {col} via IQR: {len(idxs)} flagged")
            except Exception:
                col_counts["iqr"] = None
                self._log(f"  • {col} via IQR: error")

            # Z-score
            try:
                idxs = self._zscore_outliers(series)
                col_counts["zscore"] = len(idxs)
                for i in idxs:
                    votes_df.at[i, "zscore"] = 1
                self._log(f"  • {col} via Z-score: {len(idxs)} flagged")
            except Exception:
                col_counts["zscore"] = None
                self._log(f"  • {col} via Z-score: error")

            # Modified Z
            try:
                idxs = self._modz_outliers(series)
                col_counts["modz"] = len(idxs)
                for i in idxs:
                    votes_df.at[i, "modz"] = 1
                self._log(f"  • {col} via Modified Z: {len(idxs)} flagged")
            except Exception:
                col_counts["modz"] = None
                self._log(f"  • {col} via Modified Z: error")

            # Tukey
            try:
                idxs = self._tukey_outliers(series)
                col_counts["tukey"] = len(idxs)
                for i in idxs:
                    votes_df.at[i, "tukey"] = 1
                self._log(f"  • {col} via Tukey (2×IQR): {len(idxs)} flagged")
            except Exception:
                col_counts["tukey"] = None
                self._log(f"  • {col} via Tukey: error")

            # Percentile
            try:
                idxs = self._percentile_outliers(series)
                col_counts["percentile"] = len(idxs)
                for i in idxs:
                    votes_df.at[i, "percentile"] = 1
                self._log(
                    f"  • {col} via 1st/99th percentile: {len(idxs)} flagged")
            except Exception:
                col_counts["percentile"] = None
                self._log(f"  • {col} via percentile: error")

            # Save counts
            self.report["univariate_outliers"][col] = col_counts

        # 3) MULTIVARIATE VOTING
        multi_idxs = self.detect_multivariate_outliers()
        method_used = self.report["multivariate_outliers"].get("method", None)
        self._log(
            f"  • Multivariate ({method_used}): {len(multi_idxs)} flagged")
        for i in multi_idxs:
            # Mark in the correct column
            if method_used == "Mahalanobis":
                votes_df.at[i, "mahalanobis"] = 1
            elif method_used == "LOF":
                votes_df.at[i, "lof"] = 1
            else:
                votes_df.at[i, "isolation"] = 1

        # 4) REAL OUTLIERS = rows whose sum of votes ≥ outlier_threshold
        votes_df["total_votes"] = votes_df.sum(axis=1)
        real_mask = votes_df["total_votes"] >= self.outlier_threshold
        real = votes_df.loc[real_mask].index.tolist()
        self.report["real_outliers"] = {"indices": real, "count": len(real)}
        self._log(f"  → Real outlier count = {len(real)}")

        # Keep the full votes_table_ for future inspection
        self.votes_table_ = votes_df.copy()

        # 5) TREAT OUTLIERS
        df_clean = self.df.copy()
        clipped_counts: Dict[str, int] = {}

        # Decide if we should winsorize or drop (or neither)
        # If model_family is "linear" or "bayesian" → force winsorize unless cap_outliers=False
        force_winsorize = (
            (self.model_family in ["linear", "bayesian"])
            and (self.cap_outliers is not False)
        )

        if self.cap_outliers is None and not force_winsorize:
            # “detect only” mode: do nothing to df_clean
            self.report["treatment"] = {
                "mode": "detect_only",
                "dropped_rows": [],
                "winsorized_counts": {}
            }
        else:
            # If either user specified cap_outliers=True, or model_family forces winsorize:
            if force_winsorize or (self.cap_outliers is True):
                # Winsorize: clip each numeric at its 1st/99th percentiles
                self._log(f"⚠ Winsorizing {len(real)} outlier rows.")
                for col in self.numeric_cols:
                    arr_orig = self.df[col].dropna().values
                    if arr_orig.size == 0:
                        clipped_counts[col] = 0
                        continue

                    p1, p99 = np.percentile(
                        arr_orig, [self.PCTL_LOW, self.PCTL_HIGH])
                    clipped_series = df_clean[col].clip(lower=p1, upper=p99)
                    # Count how many values actually changed
                    n_clipped = int(
                        (df_clean[col] < p1).sum() + (df_clean[col] > p99).sum())
                    clipped_counts[col] = n_clipped
                    df_clean[col] = clipped_series

                self.report["treatment"] = {
                    "mode": "winsorize",
                    "dropped_rows": [],
                    "winsorized_counts": clipped_counts
                }

            else:
                # cap_outliers is explicitly False → DROP the rows
                self._log(f"⚠ Dropping {len(real)} outlier rows.")
                df_clean.drop(index=real, inplace=True)
                df_clean.reset_index(drop=True, inplace=True)
                self.report["treatment"] = {
                    "mode": "drop",
                    "dropped_rows": real,
                    "winsorized_counts": {}
                }

        # 6) Save cleaned DataFrame + clip counts
        self.train_clean_ = df_clean.copy()
        self.clipped_counts_ = clipped_counts

        final_count = len(self.train_clean_)
        self._log(f"→ After treatment, training set has {final_count} rows.")

        return self

    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Convenience: run fit(...) then return train_clean_.
        """
        return self.fit(df, numeric_cols).train_clean_

    # ────────────── Transform (Flag Only on New Data) ──────────────

    def transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Flag outliers on new data (no dropping or winsorization). Returns a copy
        of df with an extra boolean column 'is_outlier'. Relies on:
          – self.train_clean_         (to derive univariate fences)
          – self.scaler & self.cov_estimator (for Mahalanobis)
          – self.lof_model & self.iso_model   (if used)
          – self.mahal_threshold       (χ² cutoff)
          – self.outlier_threshold     (vote threshold)
        """
        result = df.copy()
        df_num = result[numeric_cols].copy()

        # Initialize votes column
        votes = pd.Series(0, index=result.index, dtype=int)

        # Univariate fences on new data, using training fences
        def _apply_univ_fence(col: str):
            if col not in self.train_clean_.columns:
                return

            arr_train = self.train_clean_[col].dropna().values
            if arr_train.size == 0:
                return

            # (a) IQR
            q1, q3 = np.percentile(arr_train, [25, 75])
            iqr = q3 - q1
            lb_iqr, ub_iqr = q1 - self.UNIV_IQR_FACTOR * \
                iqr, q3 + self.UNIV_IQR_FACTOR * iqr
            mask_iqr = (result[col] < lb_iqr) | (result[col] > ub_iqr)
            votes.loc[mask_iqr] += 1

            # (b) Z-score
            mu, sigma = arr_train.mean(), arr_train.std(ddof=0)
            if sigma != 0 and not np.isnan(sigma):
                z = (result[col] - mu) / sigma
                mask_z = z.abs() > self.UNIV_ZSCORE_CUTOFF
                votes.loc[mask_z] += 1

            # (c) Modified Z-score
            med = np.median(arr_train)
            mad = np.median(np.abs(arr_train - med))
            if mad != 0:
                modz = 0.6745 * (result[col] - med) / mad
                mask_modz = modz.abs() > self.UNIV_MODZ_CUTOFF
                votes.loc[mask_modz] += 1

            # (d) Tukey
            lb_tukey, ub_tukey = q1 - self.TUKEY_MULTIPLIER * \
                iqr, q3 + self.TUKEY_MULTIPLIER * iqr
            mask_tukey = (result[col] < lb_tukey) | (result[col] > ub_tukey)
            votes.loc[mask_tukey] += 1

            # (e) 1st/99th percentile
            p1, p99 = np.percentile(arr_train, [self.PCTL_LOW, self.PCTL_HIGH])
            mask_pct = (result[col] < p1) | (result[col] > p99)
            votes.loc[mask_pct] += 1

        for col in numeric_cols:
            if col in result.columns:
                _apply_univ_fence(col)

        # Multivariate – Mahalanobis
        if (self.scaler is not None) and (self.cov_estimator is not None):
            complete_mask = ~df_num.isna().any(axis=1)
            if complete_mask.any():
                try:
                    Xz_new = self.scaler.transform(
                        df_num.loc[complete_mask].values)
                    md_new = self.cov_estimator.mahalanobis(Xz_new)
                    idxs_mha = df_num.loc[complete_mask].index[md_new >
                                                               self.mahal_threshold]
                    votes.loc[idxs_mha] += 1
                except Exception:
                    pass

        # Multivariate – LOF
        if self.lof_model is not None:
            complete_mask = ~df_num.isna().any(axis=1)
            if complete_mask.any():
                try:
                    preds_lof = self.lof_model.predict(
                        df_num.loc[complete_mask].values)
                    idxs_lof = df_num.loc[complete_mask].index[preds_lof == -1]
                    votes.loc[idxs_lof] += 1
                except Exception:
                    pass

        # Multivariate – IsolationForest
        if self.iso_model is not None:
            complete_mask = ~df_num.isna().any(axis=1)
            if complete_mask.any():
                try:
                    preds_iso = self.iso_model.predict(
                        df_num.loc[complete_mask].values)
                    idxs_iso = df_num.loc[complete_mask].index[preds_iso == -1]
                    votes.loc[idxs_iso] += 1
                except Exception:
                    pass

        # Final outlier flag
        result["is_outlier"] = votes >= self.outlier_threshold
        return result


# from stage3_outlier_detection import OutlierDetector

# # 1) Create detector
# detector = OutlierDetector(
#     outlier_threshold=3,
#     robust_covariance=True,
#     cap_outliers=None,        # “detect only” mode
#     model_family="linear",    # will force winsorize unless cap_outliers=False
#     random_state=0,
#     verbose=True
# )

# # 2) Fit + Treat on training data
# train_clean = detector.fit_transform(train_df, numeric_cols)

# # 3) Inspect reports if you like:
# print(detector.report["univariate_outliers"])         # per-column rule counts
# print(detector.report["multivariate_outliers"])       # chosen method, etc.
# print(detector.report["real_outliers"])               # list of flagged indices
# print(detector.report["treatment"])                   # what was done: drop/winsorize
# print(detector.votes_table_.head())                   # see each row’s votes
# print(detector.clipped_counts_)                       # how many clipped per column

# # 4) On new (validation/test) data, simply flag:
# val_flagged = detector.transform(val_df, numeric_cols)
# # – val_flagged will have the same columns + an “is_outlier” boolean column
