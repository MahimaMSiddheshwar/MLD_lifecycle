#!/usr/bin/env python3
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.utils.perfkit import perfclass, PerfMixin


@perfclass()
class OutlierDetector(BaseEstimator, TransformerMixin, PerfMixin):
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
    PCTL_LOW = 3      # 3st percentile
    PCTL_HIGH = 97     # 97th percentile

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

        self.best_rules_per_column_: Dict[str, List[str]] = {}
        self.fences_: Dict[str, Dict[str, Tuple[float, float]]] = {}
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

    def _save_state(self, filepath: str = "outlier_model_state.pkl"):
        state = {
            "numeric_cols": self.numeric_cols,
            "scaler": self.scaler,
            "cov_estimator": self.cov_estimator,
            "lof_model": self.lof_model,
            "iso_model": self.iso_model,
            "mahal_threshold": self.mahal_threshold,
            "outlier_threshold": self.outlier_threshold,
            "best_rules_per_column_": self.best_rules_per_column_,
            "fences_": self.fences_,
            "model_family": self.model_family,
            "cap_outliers": self.cap_outliers,
            "robust_covariance": self.robust_covariance,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        self._log(f"✔ Model state saved to {filepath}")

    def _load_state(self, filepath: str = "outlier_model_state.pkl"):
        if not Path(filepath).exists():
            raise RuntimeError("No model fitted. Run `.fit()` first.")
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        self._log(f"✔ Model state loaded from {filepath}")

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
        gaussian_like = ((skews < self.GAUSS_SKEW_THRESH)
                         & (kurts < self.GAUSS_KURT_THRESH)).sum()
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
        if not mahal_used and (n_samples < self.LOF_MAX_SAMPLES) and (n_features < self.LOF_MAX_FEATURES):
            self._fit_lof(df_num)
            lof_idxs = self._compute_lof_indices(df_num)
            frac_flagged = len(lof_idxs) / float(n_samples)
            if frac_flagged <= 0.05:
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

    def find_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Helper function to find numeric columns in a DataFrame.
        Returns a list of column names that are numeric.
        """
        return df.select_dtypes(include=[np.number]).columns.tolist()

    # ────────────── Main Fit & Fit_Transform ──────────────

    def fit(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = self.find_numeric_columns(df)
        self.best_rules_per_column_ = {}
        self.fences_ = {}

        votes_df = pd.DataFrame(0, index=self.df.index, columns=[
            "iqr", "zscore", "modz", "tukey", "percentile",
            "mahalanobis", "lof", "isolation"
        ], dtype=int)

        def score_rules(col: str):
            scores = {}
            fences = {}
            series = self.df[col]

            def try_rule(rule_name, func, *args):
                try:
                    idxs = func(series, *args)
                    for i in idxs:
                        votes_df.at[i, rule_name] = 1
                    scores[rule_name] = len(idxs)
                    return idxs
                except Exception:
                    scores[rule_name] = -1
                    return []

            # IQR
            idxs = try_rule("iqr", self._iqr_outliers)
            if idxs:
                q1, q3 = np.percentile(series.dropna().values, [25, 75])
                iqr = q3 - q1
                fences["iqr"] = (q1 - self.UNIV_IQR_FACTOR
                                 * iqr, q3 + self.UNIV_IQR_FACTOR * iqr)

            # Z-score
            idxs = try_rule("zscore", self._zscore_outliers)
            if idxs:
                mu, sigma = series.mean(), series.std(ddof=0)
                fences["zscore"] = (mu, sigma)

            # Modified Z
            idxs = try_rule("modz", self._modz_outliers)
            if idxs:
                med = np.median(series.dropna())
                mad = np.median(np.abs(series.dropna() - med))
                fences["modz"] = (med, mad)

            # Tukey
            idxs = try_rule("tukey", self._tukey_outliers)
            if idxs:
                q1, q3 = np.percentile(series.dropna().values, [25, 75])
                iqr = q3 - q1
                fences["tukey"] = (q1 - self.TUKEY_MULTIPLIER
                                   * iqr, q3 + self.TUKEY_MULTIPLIER * iqr)

            # Percentile
            idxs = try_rule("percentile", self._percentile_outliers)
            if idxs:
                p1, p99 = np.percentile(series.dropna().values, [
                                        self.PCTL_LOW, self.PCTL_HIGH])
                fences["percentile"] = (p1, p99)

            # Pick best 2 rules (you can limit to 1 if you want stricter)
            sorted_rules = sorted(
                [(k, v) for k, v in scores.items() if v >= 0], key=lambda x: -x[1])
            best = [r[0] for r in sorted_rules[:2]]
            return col, best, fences

        results = self.parallel_map(
            score_rules, self.numeric_cols, prefer="threads")

        for col, best_rules, fences in results:
            self.best_rules_per_column_[col] = best_rules
            self.fences_[col] = fences

        # Multivariate voting
        multi_idxs = self.detect_multivariate_outliers()
        method_used = self.report["multivariate_outliers"].get("method", None)
        for i in multi_idxs:
            if method_used == "Mahalanobis":
                votes_df.at[i, "mahalanobis"] = 1
            elif method_used == "LOF":
                votes_df.at[i, "lof"] = 1
            else:
                votes_df.at[i, "isolation"] = 1

        # Real outliers
        votes_df["total_votes"] = votes_df.sum(axis=1)
        real_mask = votes_df["total_votes"] >= self.outlier_threshold
        real = votes_df.index[real_mask].tolist()
        self.report["real_outliers"] = {"indices": real, "count": len(real)}
        self.votes_table_ = votes_df.copy()

        # Treatment
        df_clean = self.df.copy()
        clipped_counts = {}

        force_winsorize = (self.model_family in ["linear", "bayesian"]) and (
            self.cap_outliers is not False)

        if self.cap_outliers is None and not force_winsorize:
            self.train_clean_ = df_clean.copy()
            self.clipped_counts_ = clipped_counts
            self.report["treatment"] = {"mode": "detect_only"}
        elif force_winsorize or self.cap_outliers is True:
            for col in self.numeric_cols:
                arr = self.df[col].dropna().values
                if arr.size == 0:
                    continue
                p1, p99 = np.percentile(arr, [self.PCTL_LOW, self.PCTL_HIGH])
                df_clean[col] = df_clean[col].clip(lower=p1, upper=p99)
                clipped_counts[col] = int(
                    (df_clean[col] < p1).sum() + (df_clean[col] > p99).sum())

            self.train_clean_ = df_clean.copy()
            self.clipped_counts_ = clipped_counts
            self.report["treatment"] = {
                "mode": "winsorize", "counts": clipped_counts}
        else:
            df_clean.drop(index=real, inplace=True)
            df_clean.reset_index(drop=True, inplace=True)
            self.train_clean_ = df_clean.copy()
            self.clipped_counts_ = clipped_counts
            self.report["treatment"] = {"mode": "drop", "dropped_rows": real}

        self._save_state("outlier_model_state.pkl")
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience: run fit(...) then return train_clean_.
        """
        return self.fit(df).train_clean_

    # ────────────── Transform (Flag Only on New Data) ──────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._load_state("outlier_model_state.pkl")
        result = df.copy()
        numeric_cols = [col for col in self.numeric_cols if col in df.columns]
        df_num = result[numeric_cols].copy()
        votes = pd.Series(0, index=result.index, dtype=int)

        for col in numeric_cols:
            if col not in self.best_rules_per_column_:
                continue
            rules = self.best_rules_per_column_[col]
            fences = self.fences_.get(col, {})
            series = result[col]

            for rule in rules:
                try:
                    if rule == "iqr" or rule == "tukey":
                        lb, ub = fences[rule]
                        votes.loc[(series < lb) | (series > ub)] += 1
                    elif rule == "zscore":
                        mu, sigma = fences[rule]
                        if sigma != 0:
                            z = (series - mu) / sigma
                            votes.loc[z.abs() > self.UNIV_ZSCORE_CUTOFF] += 1
                    elif rule == "modz":
                        med, mad = fences[rule]
                        if mad != 0:
                            modz = 0.6745 * (series - med) / mad
                            votes.loc[modz.abs() > self.UNIV_MODZ_CUTOFF] += 1
                    elif rule == "percentile":
                        p1, p99 = fences[rule]
                        votes.loc[(series < p1) | (series > p99)] += 1
                except Exception:
                    continue

        # Multivariate
        if self.scaler and self.cov_estimator:
            mask = ~df_num.isna().any(axis=1)
            if mask.any():
                try:
                    Xz = self.scaler.transform(df_num.loc[mask].values)
                    md = self.cov_estimator.mahalanobis(Xz)
                    votes.loc[df_num.loc[mask].index[md
                                                     > self.mahal_threshold]] += 1
                except Exception:
                    pass

        if self.lof_model:
            mask = ~df_num.isna().any(axis=1)
            if mask.any():
                try:
                    preds = self.lof_model.predict(df_num.loc[mask].values)
                    votes.loc[df_num.loc[mask].index[preds == -1]] += 1
                except Exception:
                    pass

        if self.iso_model:
            mask = ~df_num.isna().any(axis=1)
            if mask.any():
                try:
                    preds = self.iso_model.predict(df_num.loc[mask].values)
                    votes.loc[df_num.loc[mask].index[preds == -1]] += 1
                except Exception:
                    pass

        self.outlier_flags_ = votes >= self.outlier_threshold
        return result.copy()
