import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from scipy import stats
from scipy.stats import chi2
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer,
    MissingIndicator
)
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer
)
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


class DataQualityPipeline:
    """
    A one‐stop pipeline from raw DataFrame → cleaned, transformed, imputed,
    (optionally) reduced DataFrame, tailored for a given “model family.”

    Steps included:
      1. Univariate outlier detection (IQR, Z‐score, Modified Z, Tukey’s fence, percentile).
      2. Multivariate outlier detection (Mahalanobis, IsolationForest, LOF).
      3. Unified “real”‐outlier filter: flagged by ≥N methods (default ≥2).
      4. Univariate transformation grid (none, log1p, box-cox, Yeo-Johnson, quantile).
      5. Missing‐value imputation grid:
         – “mean”, “median”, “most_frequent”, “constant” (0),
         – “knn” (KNNImputer), “mice” (IterativeImputer),
         – “random_sample” (random draw from empirical distribution).
      6. Scaling/normalization (Standard, Robust, MinMax) chosen by “model family.”
      7. Dimensionality reduction (optional): PCA, KernelPCA, TruncatedSVD, LDA (if classification).
      8. Final output is cleaned DataFrame + (optional) DR projections.

    The pipeline records every choice in self.report for later inspection.

    Parameters
    ----------
    df: pd.DataFrame
        Raw input. Numeric columns undergo cleaning; non‐numeric columns are untouched
        except reported. The target column (if any) should be passed explicitly.

    target_column: Optional[str]
        Name of the target column (e.g. “is_churn”). Required if you want LDA. Otherwise
        for unsupervised DR the pipeline will skip LDA.

    model_family: str
        One of {"auto", "linear", "tree", "knn", "svm", "bayesian"}, which influences:
          – default imputer preferences (mean vs median vs mice, etc.)
          – default scaler (Standard vs Robust vs MinMax)
          – whether LDA is possible (only if “linear” or “bayesian” & classification).
        If “auto”, pipeline will guess “linear” for numeric regression target,
        or “tree” if target is classification.

    outlier_methods: List[str]
        Which univariate methods to try: {"iqr", "zscore", "modz", "tukey", "percentile"}.
    multivariate_methods: List[str]
        Which multivariate methods: {"mahalanobis", "isolation_forest", "lof"}.

    transform_methods: List[str]
        Grid of transformations: {"none", "log1p", "boxcox", "yeo", "quantile"}.

    impute_methods: List[str]
        Grid of imputers: {"mean", "median", "mode", "constant_zero",
                           "knn", "mice", "random_sample"}.

    outlier_threshold: int
        Minimum number of outlier detectors (uni+multi) that must agree to flag a point.
        Default 2 (e.g. both IQR and Mahalanobis).

    pca_variance_threshold: float
        Explained‐variance threshold for selecting PCA components (default 0.90).

    verbose: bool
        If True, print step‐by‐step decisions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        model_family: str = "auto",
        outlier_methods: Optional[List[str]] = None,
        multivariate_methods: Optional[List[str]] = None,
        transform_methods: Optional[List[str]] = None,
        impute_methods: Optional[List[str]] = None,
        outlier_threshold: int = 2,
        pca_variance_threshold: float = 0.90,
        min_samples_per_feature: int = 5,
        verbose: bool = True,
    ):
        self.raw = df.copy()
        self.df = df.copy()
        self.target = target_column
        self.verbose = verbose
        self.min_samples_per_feature = min_samples_per_feature
        self.pca_variance_threshold = pca_variance_threshold
        self.outlier_threshold = outlier_threshold

        # Determine model_family if “auto”
        self.model_family = model_family.lower()
        if self.model_family == "auto":
            # Guess based on target dtypes
            if self.target and pd.api.types.is_numeric_dtype(df[self.target]):
                self.model_family = "linear"
            else:
                self.model_family = "tree"

        # Standard lists for each step
        self.outlier_methods = outlier_methods or [
            "iqr", "zscore", "modz", "tukey", "percentile"
        ]
        self.multivariate_methods = multivariate_methods or [
            "mahalanobis", "isolation_forest", "lof"
        ]
        self.transform_methods = transform_methods or [
            "none", "log1p", "boxcox", "yeo", "quantile"
        ]
        self.impute_methods = impute_methods or [
            "mean", "median", "mode", "constant_zero", "knn", "mice", "random_sample"
        ]

        # Will hold fitted DR model if any
        self._dr_model = None
        self._dr_type = None

        # Reports dictionary
        self.report: Dict[str, Dict] = {
            "univariate_outliers": {},   # {col: {method→count, …}}
            "multivariate_outliers": {},  # {method→[indices], …}
            "real_outliers": {"count": 0, "indices": []},
            "transform": {},             # {col: {"chosen":..., "scores":{…}}}
            "imputation": {},            # {col: {"chosen":…, "metrics":{…}}}
            "scaler": None,              # which scaler was used
            "dim_reduction": {},         # DR type + chosen n_components, variance
            "non_numeric": [],           # list of non‐numeric columns
        }

        # Separate numeric vs non‐numeric
        self.numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        self.non_numeric_cols = [
            c for c in self.df.columns if c not in self.numeric_cols
        ]
        self.report["non_numeric"] = self.non_numeric_cols.copy()

    # ──────────────────────── Univariate Outliers ──────────────────────────

    def _iqr_outliers(self, s: pd.Series) -> pd.Index:
        q1, q3 = np.percentile(s.dropna(), [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return s[(s < lower) | (s > upper)].index

    def _zscore_outliers(self, s: pd.Series) -> pd.Index:
        arr = s.dropna()
        mu, sigma = arr.mean(), arr.std()
        z = (arr - mu) / sigma
        return arr[np.abs(z) > 3].index

    def _modz_outliers(self, s: pd.Series) -> pd.Index:
        arr = s.dropna()
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return pd.Index([], dtype=int)
        modz = 0.6745 * (arr - median) / mad
        return arr[np.abs(modz) > 3.5].index

    def _tukey_outliers(self, s: pd.Series) -> pd.Index:
        # Tukey’s fence is same as IQR method but can also use a different multiplier
        q1, q3 = np.percentile(s.dropna(), [25, 75])
        iqr = q3 - q1
        # Tukey recommends [1.5×, 3×], we pick 2×
        lower, upper = q1 - 2.0 * iqr, q3 + 2.0 * iqr
        return s[(s < lower) | (s > upper)].index

    def _percentile_outliers(self, s: pd.Series) -> pd.Index:
        # Flag as outlier if below 1st or above 99th percentile
        p1, p99 = np.percentile(s.dropna(), [1, 99])
        return s[(s < p1) | (s > p99)].index

    def _detect_univariate_outliers(self) -> None:
        """
        Run each univariate method on every numeric column. Record counts in self.report.
        """
        for col in self.numeric_cols:
            self.report["univariate_outliers"][col] = {}
            series = self.df[col]

            for method in self.outlier_methods:
                try:
                    if method == "iqr":
                        idx = self._iqr_outliers(series)
                    elif method == "zscore":
                        idx = self._zscore_outliers(series)
                    elif method == "modz":
                        idx = self._modz_outliers(series)
                    elif method == "tukey":
                        idx = self._tukey_outliers(series)
                    elif method == "percentile":
                        idx = self._percentile_outliers(series)
                    else:
                        continue
                    self.report["univariate_outliers"][col][method] = int(
                        len(idx))
                except Exception:
                    self.report["univariate_outliers"][col][method] = None

    # ───────────────────── Multivariate Outliers ───────────────────────────

    def _mahalanobis_outliers(self) -> pd.Index:
        numeric = self.df[self.numeric_cols].dropna()
        if numeric.shape[0] < 10 or numeric.shape[1] < 2:
            return pd.Index([], dtype=int)
        cov = EmpiricalCovariance().fit(numeric.values)
        md = cov.mahalanobis(numeric.values)
        thresh = chi2.ppf(0.975, df=numeric.shape[1])
        return numeric.index[md > thresh]

    def _isolation_forest_outliers(self) -> pd.Index:
        from sklearn.ensemble import IsolationForest

        numeric = self.df[self.numeric_cols].dropna()
        if numeric.shape[0] < 50:
            return pd.Index([], dtype=int)  # too few samples
        iso = IsolationForest(contamination=0.01, random_state=0)
        mask = iso.fit_predict(numeric.values) == -1
        return numeric.index[mask]

    def _lof_outliers(self) -> pd.Index:
        from sklearn.neighbors import LocalOutlierFactor

        numeric = self.df[self.numeric_cols].dropna()
        if numeric.shape[0] < 20:
            return pd.Index([], dtype=int)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
        mask = lof.fit_predict(numeric.values) == -1
        return numeric.index[mask]

    def _detect_multivariate_outliers(self) -> None:
        """
        Run each multivariate method and store flagged indices in self.report.
        """
        for method in self.multivariate_methods:
            try:
                if method == "mahalanobis":
                    idx = self._mahalanobis_outliers()
                elif method == "isolation_forest":
                    idx = self._isolation_forest_outliers()
                elif method == "lof":
                    idx = self._lof_outliers()
                else:
                    idx = pd.Index([], dtype=int)
                self.report["multivariate_outliers"][method] = list(idx)
            except Exception:
                self.report["multivariate_outliers"][method] = []

    def _filter_real_outliers(self) -> None:
        """
        Combine univariate + multivariate flags. A data point is a “real outlier”
        if it appears in at least `outlier_threshold` of all methods combined.
        """
        # Build a dictionary: row_index → number_of_flags
        flag_counts: Dict[int, int] = {}
        # Count univariate flags
        for col, methods_dict in self.report["univariate_outliers"].items():
            for method, count in methods_dict.items():
                # We only recorded counts; need the actual indices to build flags.
                # Recompute indices for that (inefficient but keeps code DRY)
                series = self.df[col]
                if method == "iqr":
                    idxs = self._iqr_outliers(series)
                elif method == "zscore":
                    idxs = self._zscore_outliers(series)
                elif method == "modz":
                    idxs = self._modz_outliers(series)
                elif method == "tukey":
                    idxs = self._tukey_outliers(series)
                elif method == "percentile":
                    idxs = self._percentile_outliers(series)
                else:
                    idxs = pd.Index([], dtype=int)

                for i in idxs:
                    flag_counts[i] = flag_counts.get(i, 0) + 1

        # Count multivariate flags
        for method, idx_list in self.report["multivariate_outliers"].items():
            for i in idx_list:
                flag_counts[i] = flag_counts.get(i, 0) + 1

        # Any index with ≥ outlier_threshold flags is a “real outlier”
        real_idxs = [i for i, cnt in flag_counts.items() if cnt >=
                     self.outlier_threshold]
        self.report["real_outliers"]["count"] = len(real_idxs)
        self.report["real_outliers"]["indices"] = real_idxs

    def _treat_outliers(self) -> None:
        """
        Remove or Winsorize all “real outliers.” If model_family is “linear” or “bayesian,”
        prefer Winsorization (less data loss). Otherwise drop rows.
        """
        self._detect_univariate_outliers()
        self._detect_multivariate_outliers()
        self._filter_real_outliers()
        real_idxs = self.report["real_outliers"]["indices"]

        if len(real_idxs) == 0:
            if self.verbose:
                print("✔ No real outliers detected.")
            return

        if self.model_family in ["linear", "bayesian"] or self.winsorize:
            # Winsorize each numeric column at its univariate 1st/99th percentiles
            if self.verbose:
                print(
                    f"⚠ Winsorizing {len(real_idxs)} outlier rows (model_family={self.model_family})")
            for col in self.numeric_cols:
                series = self.df[col]
                p1, p99 = np.percentile(series.dropna(), [1, 99])
                self.df[col] = series.clip(lower=p1, upper=p99)
        else:
            if self.verbose:
                print(
                    f"⚠ Dropping {len(real_idxs)} outlier rows (model_family={self.model_family})")
            self.df.drop(index=real_idxs, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

    # ───────────────────────── Transformation Grid ──────────────────────────

    def _evaluate_transform(
        self, series: pd.Series
    ) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """
        For one numeric Series, try each transform in transform_methods:
          - none: original
          - log1p: np.log1p (skip if ≤ -1)
          - boxcox: boxcox (skip if any ≤ 0)
          - yeo: PowerTransformer(yeo)
          - quantile: QuantileTransformer→normal
        Score each by (shapiro_p, −|skew|). Return chosen and a dict of (shapiro_p, skew).
        """
        data = series.dropna()
        scores: Dict[str, Tuple[float, float]] = {}
        if data.empty or data.nunique() < 5:
            # Insufficient variation → no transform
            return "none", {"none": (1.0, 0.0)}

        for method in self.transform_methods:
            try:
                if method == "none":
                    arr = data.values
                elif method == "log1p":
                    if (data <= -1).any():
                        continue
                    arr = np.log1p(data.values)
                elif method == "boxcox":
                    if (data <= 0).any():
                        continue
                    arr, _ = stats.boxcox(data.values)
                elif method == "yeo":
                    pt = PowerTransformer(
                        method="yeo-johnson", standardize=True)
                    arr = pt.fit_transform(
                        data.values.reshape(-1, 1)).flatten()
                elif method == "quantile":
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=0)
                    arr = qt.fit_transform(
                        data.values.reshape(-1, 1)).flatten()
                else:
                    continue

                # Shapiro (sample ≤ 5000)
                samp = arr if arr.size <= 5000 else np.random.choice(
                    arr, 5000, replace=False)
                pval = stats.shapiro(samp)[1] if samp.size >= 3 else 0.0
                skew_abs = abs(stats.skew(arr))
                scores[method] = (pval, skew_abs)
            except Exception:
                continue

        # Choose method with max (pval, −skew_abs)
        best_method = "none"
        best_score = (-1.0, np.inf)
        for m, (pval, skew_abs) in scores.items():
            if (pval, -skew_abs) > best_score:
                best_score = (pval, -skew_abs)
                best_method = m

        return best_method, scores

    def _apply_transform(self) -> None:
        """
        Apply the chosen transform to each numeric column, record in self.report["transform"].
        """
        if self.verbose:
            print("\n🔍 Step 2: Transformation Recommendation & Application")

        for col in self.numeric_cols:
            series = self.df[col]
            best_method, scores = self._evaluate_transform(series)
            self.report["transform"][col] = {
                "chosen": best_method, "scores": scores}

            if best_method == "none":
                continue
            elif best_method == "log1p":
                self.df[col] = np.log1p(series)
            elif best_method == "boxcox":
                self.df[col], _ = stats.boxcox(
                    series.replace(0, 1).values)  # shift if needed
            elif best_method == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                self.df[col] = pt.fit_transform(
                    series.values.reshape(-1, 1)).flatten()
            elif best_method == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal", random_state=0)
                self.df[col] = qt.fit_transform(
                    series.values.reshape(-1, 1)).flatten()

            if self.verbose:
                print(f"  • {col:20s} → {best_method}")

    # ───────────────────────── Missing‐Value Imputation ──────────────────────

    def _random_sample_impute(self, orig: pd.Series) -> pd.Series:
        """
        Fill missing in orig by sampling from non‐missing empirical distribution.
        """
        nonnull = orig.dropna().values
        full = orig.copy()
        mask = full.isna()
        full.loc[mask] = np.random.choice(
            nonnull, size=mask.sum(), replace=True)
        return full

    def _evaluate_imputation(
        self,
        col: str,
        orig: pd.Series,
        imputed: pd.Series,
        cov_before: np.ndarray,
        cov_after: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        For column `col`, compute:
          - ks_p: KS‐test p‐value between orig_nonnull vs imputed_nonnull
          - var_ratio: var(imputed_nonnull) / var(orig_nonnull)
          - cov_change: relative change in covariance row for `col`
        """
        orig_nonnull = orig.dropna().values
        imp_nonnull = imputed.dropna().values
        ks_p = 0.0
        if orig_nonnull.size >= 2 and imp_nonnull.size >= 2:
            try:
                ks_p = stats.ks_2samp(orig_nonnull, imp_nonnull)[1]
            except Exception:
                ks_p = 0.0

        var_orig = np.nanvar(orig_nonnull)
        var_imp = np.nanvar(imp_nonnull)
        var_ratio = var_imp / var_orig if var_orig > 0 else np.nan

        idx = self.numeric_cols.index(col)
        diff = np.abs(cov_after[idx, :] - cov_before[idx, :])
        denom = np.abs(cov_before[idx, :])
        cov_change = np.sum(diff / (denom + 1e-9))

        return ks_p, var_ratio, cov_change

    def _impute_column(
        self, df_num: pd.DataFrame, col: str, cov_before: np.ndarray
    ) -> Tuple[str, pd.Series, Dict[str, Tuple[float, float, float]]]:
        """
        Try each impute method on `col`. Evaluate by KS, var_ratio, cov_change.
        Choose best that also satisfies (var_ratio ≥ threshold, cov_change ≤ threshold).
        If none qualify, fall back to “mean.”
        Returns (chosen_method, imputed_series, metrics_for_all).
        """
        orig = df_num[col]
        if orig.isna().sum() == 0:
            return "none", orig.copy(), {"none": (1.0, 1.0, 0.0)}

        results: Dict[str, Tuple[float, float, float]] = {}
        candidates: Dict[str, pd.Series] = {}

        for method in self.impute_methods:
            try:
                if method == "mean":
                    imp = SimpleImputer(strategy="mean")
                    arr = pd.Series(
                        imp.fit_transform(
                            orig.values.reshape(-1, 1)).flatten(),
                        index=orig.index
                    )
                elif method == "median":
                    imp = SimpleImputer(strategy="median")
                    arr = pd.Series(
                        imp.fit_transform(
                            orig.values.reshape(-1, 1)).flatten(),
                        index=orig.index
                    )
                elif method == "mode":
                    imp = SimpleImputer(strategy="most_frequent")
                    arr = pd.Series(
                        imp.fit_transform(
                            orig.values.reshape(-1, 1)).flatten(),
                        index=orig.index
                    )
                elif method == "constant_zero":
                    imp = SimpleImputer(strategy="constant", fill_value=0)
                    arr = pd.Series(
                        imp.fit_transform(
                            orig.values.reshape(-1, 1)).flatten(),
                        index=orig.index
                    )
                elif method == "knn":
                    imputer = KNNImputer(n_neighbors=5)
                    temp = df_num.copy()
                    temp_imp = pd.DataFrame(
                        imputer.fit_transform(temp), columns=df_num.columns, index=df_num.index
                    )
                    arr = temp_imp[col]
                elif method == "mice":
                    imputer = IterativeImputer(
                        estimator=BayesianRidge(), sample_posterior=True, random_state=0, max_iter=10
                    )
                    temp = df_num.copy()
                    temp_imp = pd.DataFrame(
                        imputer.fit_transform(temp), columns=df_num.columns, index=df_num.index
                    )
                    arr = temp_imp[col]
                elif method == "random_sample":
                    arr = self._random_sample_impute(orig)
                else:
                    continue

                # Compute covariance after
                nonnull_idx = arr.dropna().index
                if len(nonnull_idx) < self.min_samples_per_feature:
                    cov_after = cov_before.copy()
                else:
                    cov_after = EmpiricalCovariance().fit(
                        df_num.loc[nonnull_idx].assign(
                            **{col: arr.loc[nonnull_idx]}).values
                    ).covariance_

                ks_p, var_ratio, cov_chg = self._evaluate_imputation(
                    col, orig, arr, cov_before, cov_after
                )

                results[method] = (ks_p, var_ratio, cov_chg)
                candidates[method] = arr

            except Exception:
                continue

        # Select best by (ks_p, var_ratio descending, cov_chg ascending)
        best_method = None
        best_score = (-1.0, -1.0, np.inf)
        best_arr = orig.copy()
        for m, (ks_p, var_ratio, cov_chg) in results.items():
            if (
                ks_p > best_score[0]
                and var_ratio >= self.drop_threshold_variance
                and cov_chg <= self.cov_change_threshold
            ):
                best_method = m
                best_score = (ks_p, var_ratio, cov_chg)
                best_arr = candidates[m]

        if best_method is None:
            # fallback to mean
            imp = SimpleImputer(strategy="mean")
            fallback = pd.Series(
                imp.fit_transform(orig.values.reshape(-1, 1)).flatten(), index=orig.index
            )
            ks_p_fb, var_ratio_fb, _ = self._evaluate_imputation(
                col, orig, fallback, cov_before, cov_before
            )
            self.report["imputation"][col] = {
                "chosen": "fallback_mean",
                "metrics": {"ks_p": ks_p_fb, "var_ratio": var_ratio_fb, "cov_change": np.nan},
                "note": "All methods failed QC"
            }
            return "fallback_mean", fallback, results

        self.report["imputation"][col] = {
            "chosen": best_method,
            "metrics": {"ks_p": best_score[0], "var_ratio": best_score[1], "cov_change": best_score[2]}
        }
        return best_method, best_arr, results

    def _impute_missing(self) -> None:
        """
        Impute every numeric column with QC. Record decisions in self.report["imputation"].
        """
        if self.verbose:
            print("\n🔍 Step 4: Missing‐Value Imputation & QC")

        df_num = self.df[self.numeric_cols].copy()
        complete_before = df_num.dropna()
        if complete_before.shape[0] < self.min_samples_per_feature:
            cov_before = np.zeros(
                (len(self.numeric_cols), len(self.numeric_cols)))
            if self.verbose:
                print("  • Skipping covariance QC (not enough complete cases).")
        else:
            cov_before = EmpiricalCovariance().fit(complete_before.values).covariance_

        for col in self.numeric_cols:
            orig = df_num[col]
            if orig.isna().sum() == 0:
                self.report["imputation"][col] = {
                    "chosen": "none",
                    "metrics": {"ks_p": 1.0, "var_ratio": 1.0, "cov_change": 0.0}
                }
                continue

            method, arr, _ = self._impute_column(df_num, col, cov_before)
            self.df[col] = arr
            info = self.report["imputation"][col]
            if self.verbose:
                print(
                    f"  • {col:20s} → {info['chosen']}; "
                    f"KS_p={info['metrics']['ks_p']:.3f}, "
                    f"var_ratio={info['metrics']['var_ratio']:.3f}, "
                    f"cov_change={info['metrics']['cov_change']:.3f}"
                )

    # ────────────────────── Scaling / Normalization ─────────────────────────

    def _select_scaler(self) -> None:
        """
        Choose a scaler based on model_family:
          – linear, bayesian, svm → StandardScaler
          – tree → None (trees prefer raw)
          – knn → MinMaxScaler (kNN sensitive to scale)
          – others → RobustScaler as fallback
        """
        if self.model_family in ["linear", "bayesian", "svm"]:
            scaler = StandardScaler()
            label = "StandardScaler"
        elif self.model_family == "knn":
            scaler = MinMaxScaler()
            label = "MinMaxScaler"
        elif self.model_family == "tree":
            scaler = None
            label = "None"
        else:
            scaler = RobustScaler()
            label = "RobustScaler"

        self.report["scaler"] = label
        if scaler is not None:
            self.df[self.numeric_cols] = scaler.fit_transform(
                self.df[self.numeric_cols])

    # ───────────────────── Dimensionality Reduction ──────────────────────────

    def _validate_for_dr(self) -> Tuple[np.ndarray, str]:
        """
        Ensure DR is possible:
          1) No NaNs in numeric portion
          2) Enough samples: n_samples ≥ min_samples_per_feature × n_features
          3) Numeric dims ≥ 2
          4) If LDA and classification target, ensure target is categorical
        Returns (X_values, error_message). If error_message non‐empty, skip DR.
        """
        X = self.df[self.numeric_cols].copy()
        if X.isna().any().any():
            return X.values, "DR aborted: NaNs remain after imputation."

        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < self.min_samples_per_feature * n_feats:
            return X.values, (
                f"DR aborted: insufficient samples ({n_samples}) for features ({n_feats})."
            )

        # If user requested LDA but no valid classification target, abort
        if self._dr_type == "lda":
            if self.target is None or self.target not in self.df.columns:
                return X.values, "LDA aborted: no target column provided."
            if not pd.api.types.is_categorical_dtype(self.df[self.target]) and not pd.api.types.is_object_dtype(self.df[self.target]):
                return X.values, "LDA aborted: target must be categorical for LDA."

        return X.values, ""

    def _apply_dr(self, X: np.ndarray) -> None:
        """
        Fit the chosen DR model (PCA/KPCA/SVD/LDA), compute:
          – n_components to reach pca_variance_threshold (only for PCA)
          – explained‐variance (if available)
        """
        if self._dr_type == "pca":
            self._apply_pca(X)
        elif self._dr_type == "kpca":
            self._apply_kernel_pca(X)
        elif self._dr_type == "tsvd":
            self._apply_tsvd(X)
        elif self._dr_type == "lda":
            self._apply_lda(X)
        else:
            return

    def _apply_pca(self, X: np.ndarray) -> None:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_std)
        explained = np.cumsum(pca.explained_variance_ratio_)
        n_comp = int(np.searchsorted(
            explained, self.pca_variance_threshold) + 1)

        self._dr_model = pca
        self.report["dim_reduction"] = {
            "type": "PCA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
            "explained_variance_ratio": explained.tolist(),
        }
        if self.verbose:
            print(
                f"\n🔍 DR: PCA chosen {n_comp} comps ({explained[n_comp-1]:.3f} var)")

    def _apply_kernel_pca(self, X: np.ndarray) -> None:
        # We choose RBF kernel with gamma=1/n_features by default
        gamma = 1.0 / X.shape[1]
        kpca = KernelPCA(kernel="rbf", gamma=gamma,
                         fit_inverse_transform=False)
        kpca.fit(X)
        # We cannot easily get explained variance from KernelPCA,
        # so just record that it ran successfully.
        self._dr_model = kpca
        self.report["dim_reduction"] = {
            "type": "KernelPCA",
            "params": {"kernel": "rbf", "gamma": gamma},
        }
        if self.verbose:
            print(f"\n🔍 DR: KernelPCA (rbf gamma={gamma:.3f}) fitted")

    def _apply_tsvd(self, X: np.ndarray) -> None:
        # TruncatedSVD on standardized data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        # Choose number of components = min(n_samples-1, n_features)
        n_comp = min(X_std.shape[0] - 1, X_std.shape[1])
        tsvd = TruncatedSVD(n_components=n_comp)
        tsvd.fit(X_std)
        explained = np.cumsum(tsvd.explained_variance_ratio_)
        chosen = int(np.searchsorted(
            explained, self.pca_variance_threshold) + 1)

        self._dr_model = tsvd
        self.report["dim_reduction"] = {
            "type": "TruncatedSVD",
            "n_original_features": X.shape[1],
            "n_components": chosen,
            "explained_variance_ratio": explained.tolist(),
        }
        if self.verbose:
            print(
                f"\n🔍 DR: TruncatedSVD chosen {chosen} comps ({explained[chosen-1]:.3f} var)")

    def _apply_lda(self, X: np.ndarray) -> None:
        y = self.df[self.target]
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        # LDA n_components = min(n_classes - 1, n_features)
        n_comp = min(len(np.unique(y)) - 1, X.shape[1])
        self._dr_model = lda
        self.report["dim_reduction"] = {
            "type": "LDA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
        }
        if self.verbose:
            print(
                f"\n🔍 DR: LDA chosen {n_comp} comps (n_classes={len(np.unique(y))})")

    # ───────────────────────── Main Pipeline ─────────────────────────────────

    def run(self, dr_type: Optional[str] = "pca") -> pd.DataFrame:
        """
        Execute the full pipeline in order:
          1) Outlier detection & treatment
          2) Transformation recommendation & application
          3) Missing‐value imputation & QC
          4) Scaling/normalization per model_family
          5) Dimensionality reduction (choose among PCA/KPCA/SVD/LDA), if requested

        dr_type: str or None, default "pca"
          One of {"pca", "kpca", "tsvd", "lda"} or None to skip DR entirely.
        """
        # 1) Outliers
        self._treat_outliers()

        # 2) Transform
        self._apply_transform()

        # 3) Impute missing
        self._impute_missing()

        # 4) Scale
        self._select_scaler()

        # 5) Dimensionality reduction
        if dr_type:
            self._dr_type = dr_type.lower()
            X_vals, error = self._validate_for_dr()
            if error:
                raise RuntimeError(error)
            self._apply_dr(X_vals)

        return self.df.copy()

    def print_report(self) -> None:
        """
        Print a summary of decisions for each step.
        """
        print("\n" + "=" * 80)
        print("✅ DataQualityPipeline Report")
        print("=" * 80)

        # Univariate outliers
        print("\n▶ Univariate Outliers:")
        for col, methods_dict in self.report["univariate_outliers"].items():
            line = "  • " + col + ": " + ", ".join(
                f"{m}={count}" for m, count in methods_dict.items() if count is not None
            )
            print(line if methods_dict else f"  • {col}: None")

        # Multivariate outliers
        print("\n▶ Multivariate Outliers:")
        for method, idxs in self.report["multivariate_outliers"].items():
            print(f"  • {method}: count={len(idxs)}")

        # Real outliers
        ro = self.report["real_outliers"]
        print(
            f"\n▶ Real Outliers (≥{self.outlier_threshold} flags): count={ro['count']}")

        # Transformation
        print("\n▶ Transformation Summary:")
        for col, info in self.report["transform"].items():
            print(f"  • {col:20s} → {info['chosen']}")

        # Imputation
        print("\n▶ Imputation Summary:")
        for col, info in self.report["imputation"].items():
            ch = info["chosen"]
            m = info["metrics"]
            print(
                f"  • {col:20s} → {ch}, KS_p={m['ks_p']:.3f}, var_ratio={m['var_ratio']:.3f}, cov_change={m['cov_change']:.3f}")

        # Scaler
        print(f"\n▶ Scaler: {self.report['scaler']}")

        # Dimensionality Reduction
        print("\n▶ Dimensionality Reduction:")
        dr = self.report.get("dim_reduction", {})
        if dr:
            dtype = dr["type"]
            nc = dr.get("n_components", None)
            if dtype == "PCA" or dtype == "TruncatedSVD":
                ev = dr["explained_variance_ratio"][nc - 1] if nc else None
                print(
                    f"  • {dtype}: n_original={dr['n_original_features']}, chosen_n={nc}, cumulative_var={ev:.3f}")
            elif dtype == "KernelPCA":
                print(f"  • KernelPCA: parameters={dr['params']}")
            elif dtype == "LDA":
                print(
                    f"  • LDA: n_original={dr['n_original_features']}, chosen_n={nc}")
        else:
            print("  • None")

        # Non‐numeric
        print("\n▶ Non‐numeric columns (unchanged):")
        if self.non_numeric_cols:
            print("  • " + ", ".join(self.non_numeric_cols))
        else:
            print("  • None")

        print("\n" + "=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (as a script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataQualityPipeline Demo")
    parser.add_argument(
        "--input_csv", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--target", required=False, help="Target column (for LDA, etc.)"
    )
    parser.add_argument(
        "--model_family",
        choices=["auto", "linear", "tree", "knn", "svm", "bayesian"],
        default="auto",
        help="Choose pipeline behavior for scaling/imputers",
    )
    parser.add_argument(
        "--dr_type",
        choices=["pca", "kpca", "tsvd", "lda", "none"],
        default="pca",
        help="Which dimensionality reduction to attempt (or none).",
    )
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input_csv)
    pipeline = DataQualityPipeline(
        df=df_raw,
        target_column=args.target,
        model_family=args.model_family,
        verbose=True,
    )

    try:
        df_processed = pipeline.run(
            dr_type=(None if args.dr_type == "none" else args.dr_type))
    except RuntimeError as e:
        print(f"⚠ Dimensionality Reduction skipped: {e}")
    finally:
        pipeline.print_report()

    df_processed.to_csv("processed_output.csv", index=False)
    print("✨ Cleaned data written to 'processed_output.csv'")
