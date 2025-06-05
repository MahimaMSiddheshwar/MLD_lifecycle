#!/usr/bin/env python3
"""
stage3_outlier_detection.py

– Univariate outliers: IQR, Z‐score, Modified Z
– Multivariate outliers: Mahalanobis on scaled data, with robust fallback
– Vote‐based: if a row is flagged by ≥ threshold methods, it is a “real outlier”
– Optionally cap outliers instead of drop (configurable)
– Provides transform() to flag (but not drop) outliers on new data

Usage:
    from stage3_outlier_detection import OutlierDetector
    detector = OutlierDetector(
        univ_flag_threshold=2,
        robust_covariance=True,
        cap_outliers=False,
        verbose=True
    )
    train_clean = detector.fit_transform(train_df, numeric_cols)
    val_flagged = detector.transform(val_df, numeric_cols)
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierDetector(BaseEstimator, TransformerMixin):
    """
    Detect and remove/cap outliers:
      – Univariate: IQR, Z‐score, Modified Z
      – Multivariate: Mahalanobis on z‐scored numerics, robust fallback
      – Voting: real outlier if votes ≥ univ_flag_threshold
      – Optionally cap outliers at nearest non‐outlier boundary
      – transform() flags outliers on new data (no removal by default)
    """

    def __init__(
        self,
        univ_flag_threshold: int = 2,
        robust_covariance: bool = True,
        cap_outliers: bool = False,
        verbose: bool = False,
    ):
        self.univ_flag_threshold = univ_flag_threshold
        self.robust_covariance = robust_covariance
        self.cap_outliers = cap_outliers
        self.verbose = verbose

        # Will be set during fit
        self.numeric_cols = []
        self.scaler = None
        self.cov_estimator = None
        self.mahal_threshold = None
        self.outlier_indices_ = []    # indices dropped or flagged in train
        self.votes_table_ = None      # DataFrame of vote counts per row

        # Reporting
        self.report = {
            "univariate": {},
            "multivariate": {},
            "real_outliers": {},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ——— Univariate outlier methods ———

    def _iqr_outliers(self, series: pd.Series):
        arr = series.dropna().values
        if len(arr) == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return series[(series < lb) | (series > ub)].index.tolist()

    def _zscore_outliers(self, series: pd.Series):
        arr = series.dropna().values
        if len(arr) < 2:
            return []
        mu, sigma = series.mean(), series.std()
        if sigma == 0 or np.isnan(sigma):
            return []
        z = (series - mu) / sigma
        return series[np.abs(z) > 3.0].index.tolist()

    def _modz_outliers(self, series: pd.Series):
        arr = series.dropna().values
        if len(arr) < 2:
            return []
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0:
            return []
        modz = 0.6745 * (series - med) / mad
        return series[np.abs(modz) > 3.5].index.tolist()

    # ——— Multivariate Mahalanobis ———

    def _fit_mahalanobis(self, df_numeric: pd.DataFrame):
        """
        Fit covariance on z‐scored data. If robust_covariance=True, try MinCovDet.
        Sets self.cov_estimator and self.mahal_threshold.
        """
        # 1) z‐score the numeric block
        scaler = StandardScaler()
        Xz = scaler.fit_transform(df_numeric.values)
        self.scaler = scaler

        # 2) Fit EmpiricalCovariance
        try:
            emp_cov = EmpiricalCovariance().fit(Xz)
            self.cov_estimator = emp_cov
        except Exception:
            self.cov_estimator = None

        # 3) If robust requested and EmpiricalCovariance yields too many flags later, fallback
        # We'll check that after computing MDs.

        # 4) Compute threshold for MD
        p = Xz.shape[1]
        self.mahal_threshold = chi2.ppf(0.975, df=p)
        return

    def _compute_mahalanobis_indices(self, df_numeric: pd.DataFrame):
        """
        Using self.scaler + self.cov_estimator, compute MDs & return indices exceeding threshold.
        If too many flagged (>5%), and robust_covariance=True, refit with MinCovDet.
        """
        Xz = self.scaler.transform(df_numeric.values)
        if self.cov_estimator is None:
            return []

        md = self.cov_estimator.mahalanobis(Xz)
        indices = df_numeric.index[md > self.mahal_threshold].tolist()
        frac_flagged = len(indices) / df_numeric.shape[0]

        if self.robust_covariance and frac_flagged > 0.05:
            # Retry with MinCovDet
            try:
                mcd = MinCovDet().fit(Xz)
                md2 = mcd.mahalanobis(Xz)
                self.cov_estimator = mcd
                indices = df_numeric.index[md2 > self.mahal_threshold].tolist()
            except Exception:
                pass

        return indices

    # ——— Main Fit_Transform ———

    def fit(self, df: pd.DataFrame, numeric_cols: list):
        """
        Fit on df (DataFrame) using numeric_cols:
          – Detect univariate outliers (IQR, Z‐score, Modified Z)
          – Detect multivariate outliers (Mahalanobis on z‐scored data)
          – Vote per row; real outliers = rows with votes ≥ univ_flag_threshold
          – If cap_outliers: record boundary values for capping
          – Else: drop these rows from df
        """
        self.numeric_cols = numeric_cols.copy()
        df_num = df[self.numeric_cols].copy()

        # 1) Univariate voting
        votes = {}
        self.report["univariate"] = {col: {} for col in self.numeric_cols}

        for col in self.numeric_cols:
            s = df[col]
            self.report["univariate"][col] = {}

            for name, fn in [
                ("iqr", self._iqr_outliers),
                ("zscore", self._zscore_outliers),
                ("modz", self._modz_outliers),
            ]:
                try:
                    idxs = fn(s)
                    self.report["univariate"][col][name] = len(idxs)
                    self._log(f"  • {col} via {name}: {len(idxs)} flagged")
                    for i in idxs:
                        votes[i] = votes.get(i, 0) + 1
                except Exception:
                    self.report["univariate"][col][name] = None
                    self._log(f"  • {col} via {name}: error")

        # 2) Multivariate (Mahalanobis)
        self.report["multivariate"] = {}
        self._fit_mahalanobis(df_num.dropna())
        maha_idxs = self._compute_mahalanobis_indices(df_num.dropna())
        self.report["multivariate"]["mahalanobis"] = maha_idxs
        self._log(f"  • Mahalanobis: {len(maha_idxs)} flagged")
        for i in maha_idxs:
            votes[i] = votes.get(i, 0) + 1

        # 3) Real outliers = votes ≥ threshold
        real = [idx for idx, v in votes.items() if v >=
                self.univ_flag_threshold]
        self.report["real_outliers"] = {"indices": real, "count": len(real)}
        self._log(f"  → Real outlier count = {len(real)}")

        self.outlier_indices_ = real

        # 4) Drop or cap
        df_clean = df.copy()
        if self.cap_outliers:
            # For each numeric column, cap values outside [lb, ub] at the nearest bound
            for col in self.numeric_cols:
                arr = df_clean[col].dropna().values
                if len(arr) == 0:
                    continue
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                df_clean[col] = df_clean[col].clip(lower=lb, upper=ub)
            self._log("  • Capped outliers at IQR bounds (no rows dropped)")
        else:
            # Drop rows
            df_clean.drop(index=real, inplace=True)
            df_clean.reset_index(drop=True, inplace=True)
            self._log(f"  • Dropped {len(real)} outlier rows")

        self.train_clean_ = df_clean.copy()
        return self

    def fit_transform(self, df: pd.DataFrame, numeric_cols: list):
        """
        Fit on (df, numeric_cols) and return the cleaned df (either capped or dropped).
        """
        return self.fit(df, numeric_cols).train_clean_

    # ——— Transform on New Data (Flag only) ———

    def transform(self, df: pd.DataFrame, numeric_cols: list):
        """
        Apply outlier‐flagging logic to new data (no dropping).
        Returns a DataFrame with a boolean column 'is_outlier' marking flagged rows.
        """
        result = df.copy()
        df_num = result[numeric_cols].copy()

        # 1) Initialize vote=0
        votes = pd.Series(0, index=df.index)

        # 2) Univariate: use same IQR/Z/ModZ thresholds computed on train
        for col in numeric_cols:
            # compute train‐side quartiles/iqr/etc. for threshold
            arr_train = self.train_clean_[col].dropna().values
            if len(arr_train) == 0:
                continue
            q1, q3 = np.percentile(arr_train, [25, 75])
            iqr = q3 - q1
            lb_iqr, ub_iqr = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            # IQR
            idxs_iqr = result[(result[col] < lb_iqr) | (
                result[col] > ub_iqr)].index.tolist()
            for i in idxs_iqr:
                votes.at[i] += 1

            # Z‐score
            mu, sigma = arr_train.mean(), arr_train.std()
            if sigma != 0 and not np.isnan(sigma):
                z = (result[col] - mu) / sigma
                idxs_z = result[np.abs(z) > 3.0].index.tolist()
                for i in idxs_z:
                    votes.at[i] += 1

            # Modified Z
            med = np.median(arr_train)
            mad = np.median(np.abs(arr_train - med))
            if mad != 0:
                modz = 0.6745 * (result[col] - med) / mad
                idxs_modz = result[np.abs(modz) > 3.5].index.tolist()
                for i in idxs_modz:
                    votes.at[i] += 1

        # 3) Multivariate: Mahalanobis on z‐scored new data
        if self.scaler is not None and self.cov_estimator is not None:
            Xz_new = self.scaler.transform(df_num.values)
            try:
                md_new = self.cov_estimator.mahalanobis(Xz_new)
                idxs_m = result.index[md_new > self.mahal_threshold].tolist()
                for i in idxs_m:
                    votes.at[i] += 1
            except Exception:
                pass

        # 4) Mark as outlier if votes ≥ threshold
        result["is_outlier"] = votes >= self.univ_flag_threshold
        return result
