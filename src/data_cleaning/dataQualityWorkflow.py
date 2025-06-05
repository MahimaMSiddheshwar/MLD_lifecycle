import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from scipy import stats
from scipy.stats import chi2
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer
)
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    OrdinalEncoder
)
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


class DataQualityPipeline:
    """
    A unified pipeline that cleans, imputes, transforms, encodes, and optionally
    applies dimensionality reduction to both NUMERIC and CATEGORICAL columns in a DataFrame.

    Steps:
      1. Numeric outliers → treat (IQR, Z‐score, Modified Z, etc.)  
      2. Numeric transformations → choose among (none, log1p, boxcox, yeo, quantile)  
      3. Numeric missing‐value imputation → try (mean, median, knn, mice, random_sample)  
      4. Numeric scaling → Standard / MinMax / Robust (based on model_family)  
      5. Dimensionality reduction → PCA / KernelPCA / TruncatedSVD / LDA

      6. Categorical missing‐value imputation → try (mode, constant_other, random_sample)  
      7. Rare‐category grouping → any category < `rare_thresh` → “__RARE__”  
      8. Categorical encoding → choose among (one‐hot, ordinal, frequency, target)  
          - “target” encoding uses the training‐set relationship to the target_column

    All decisions are recorded in self.report under keys:
      - "univariate_outliers", "multivariate_outliers", "real_outliers"  
      - "transform" (numeric)  
      - "imputation" (numeric)  
      - "scaler"  
      - "dim_reduction"  
      - "categorical": { col: { "missing_imputer":…, "rare_grouped":…, "encoder":… } }  
      - "non_numeric": list of non‐numeric columns (for reference)

    Parameters
    ----------
    df: pd.DataFrame
        Raw input.  
    target_column: Optional[str]
        Name of the target (for LDA and target encoding).  
    model_family: str
        One of {"auto", "linear", "tree", "knn", "svm", "bayesian"}.  
        “auto” → “linear” if numeric target, else “tree”.  
    outlier_methods: List[str]
        Univariate outlier detectors to try.  
    multivariate_methods: List[str]
        Multivariate outlier detectors to try.  
    transform_methods: List[str]
        Candidate numeric transforms.  
    impute_methods_num: List[str]
        Candidate numeric imputers.  
    impute_methods_cat: List[str]
        Candidate categorical imputers (mode/constant/random).  
    encode_methods: List[str]
        Candidate categorical encoders: one‐hot, ordinal, frequency, target.  
    rare_thresh: float
        Any category with relative frequency < rare_thresh is grouped into "__RARE__".  
    outlier_threshold: int
        Minimum number of flags (uni+multi) to call a row a “real outlier”.  
    pca_variance_threshold: float
        Explained‐variance threshold for choosing number of PCA components (default 0.90).  
    min_samples_per_feature: int
        Minimum complete‐case rows required to compute a reliable covariance (default 5).  
    verbose: bool
        If True, print progress and decisions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        model_family: str = "auto",
        outlier_methods: Optional[List[str]] = None,
        multivariate_methods: Optional[List[str]] = None,
        transform_methods: Optional[List[str]] = None,
        impute_methods_num: Optional[List[str]] = None,
        impute_methods_cat: Optional[List[str]] = None,
        encode_methods: Optional[List[str]] = None,
        rare_thresh: float = 0.01,
        outlier_threshold: int = 2,
        pca_variance_threshold: float = 0.90,
        min_samples_per_feature: int = 5,
        verbose: bool = True,
    ):
        self.raw = df.copy()
        self.df = df.copy()
        self.target = target_column
        self.verbose = verbose
        self.rare_thresh = rare_thresh
        self.outlier_threshold = outlier_threshold
        self.pca_variance_threshold = pca_variance_threshold
        self.min_samples_per_feature = min_samples_per_feature

        # Determine model_family if “auto”
        self.model_family = model_family.lower()
        if self.model_family == "auto":
            if self.target and pd.api.types.is_numeric_dtype(df[self.target]):
                self.model_family = "linear"
            else:
                self.model_family = "tree"

        # Set default methods
        self.outlier_methods = outlier_methods or [
            "iqr", "zscore", "modz", "tukey", "percentile"
        ]
        self.multivariate_methods = multivariate_methods or [
            "mahalanobis", "isolation_forest", "lof"
        ]
        self.transform_methods = transform_methods or [
            "none", "log1p", "boxcox", "yeo", "quantile"
        ]
        self.impute_methods_num = impute_methods_num or [
            "mean", "median", "knn", "mice", "random_sample"
        ]
        self.impute_methods_cat = impute_methods_cat or [
            "mode", "constant_other", "random_sample"
        ]
        self.encode_methods = encode_methods or [
            "onehot", "ordinal", "frequency", "target"
        ]

        # Placeholder for DR model
        self._dr_model = None
        self._dr_type = None

        # Reports
        self.report: Dict[str, Dict] = {
            "univariate_outliers": {},     # {col: {method: count, …}}
            "multivariate_outliers": {},   # {method: [indices], …}
            "real_outliers": {"count": 0, "indices": []},
            "transform": {},               # {col: {"chosen":…, "scores":{…}}}
            "imputation": {},              # {col: {"chosen":…, "metrics":{…}}}
            "scaler": None,                # name of scaler used
            "dim_reduction": {},           # DR details
            # {col: {"missing_imputer":…, "rare_grouped":…, "encoder":…}}
            "categorical": {},
            "non_numeric": [],             # list of non‐numeric columns
        }

        # Split numeric vs. categorical vs. others
        self.numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        # Everything else we treat as “categorical” (incl. object, category, boolean)
        self.categorical_cols = [
            c for c in self.df.columns if c not in self.numeric_cols and c != self.target
        ]
        self.non_numeric_cols = self.categorical_cols.copy()
        self.report["non_numeric"] = self.non_numeric_cols.copy()

    # ─────────────────────── Univariate Outliers ──────────────────────────

    def _iqr_outliers(self, s: pd.Series) -> pd.Index:
        """Flag points outside (Q1−1.5*IQR, Q3+1.5*IQR)."""
        arr = s.dropna()
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return arr[(arr < lower) | (arr > upper)].index

    def _zscore_outliers(self, s: pd.Series) -> pd.Index:
        """Flag |z| > 3."""
        arr = s.dropna()
        mu, sigma = arr.mean(), arr.std()
        z = (arr - mu) / sigma
        return arr[np.abs(z) > 3].index

    def _modz_outliers(self, s: pd.Series) -> pd.Index:
        """Flag Modified Z‐score (|0.6745*(x−med)/MAD| > 3.5)."""
        arr = s.dropna()
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return pd.Index([], dtype=int)
        modz = 0.6745 * (arr - median) / mad
        return arr[np.abs(modz) > 3.5].index

    def _tukey_outliers(self, s: pd.Series) -> pd.Index:
        """Use a 2× IQR fence (Tukey’s recommendation)."""
        arr = s.dropna()
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 2.0 * iqr, q3 + 2.0 * iqr
        return arr[(arr < lower) | (arr > upper)].index

    def _percentile_outliers(self, s: pd.Series) -> pd.Index:
        """Flag points below 1st or above 99th percentile."""
        arr = s.dropna()
        p1, p99 = np.percentile(arr, [1, 99])
        return arr[(arr < p1) | (arr > p99)].index

    def _detect_univariate_outliers(self) -> None:
        """
        For each numeric column, run every univariate method and record counts in
        self.report["univariate_outliers"][col][method] = count.
        """
        for col in self.numeric_cols:
            self.report["univariate_outliers"][col] = {}
            series = self.df[col]
            for method in self.outlier_methods:
                try:
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
                    self.report["univariate_outliers"][col][method] = int(
                        len(idxs))
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
            return pd.Index([], dtype=int)
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
        For each multivariate method, record flagged indices under
        self.report["multivariate_outliers"][method] = list_of_indices.
        """
        for method in self.multivariate_methods:
            try:
                if method == "mahalanobis":
                    idxs = self._mahalanobis_outliers()
                elif method == "isolation_forest":
                    idxs = self._isolation_forest_outliers()
                elif method == "lof":
                    idxs = self._lof_outliers()
                else:
                    idxs = pd.Index([], dtype=int)
                self.report["multivariate_outliers"][method] = list(idxs)
            except Exception:
                self.report["multivariate_outliers"][method] = []

    def _filter_real_outliers(self) -> None:
        """
        Build a dictionary of row_index → number_of_flags (uni+multi). Any row
        with ≥ self.outlier_threshold flags is marked a “real outlier.”
        """
        flag_counts: Dict[int, int] = {}
        # Count univariate flags
        for col, methods_dict in self.report["univariate_outliers"].items():
            for method, count in methods_dict.items():
                if count is None or count == 0:
                    continue
                # Re‐compute indices for that method (so we know exactly which rows)
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

        # Rows that meet threshold
        real_idxs = [i for i, cnt in flag_counts.items() if cnt >=
                     self.outlier_threshold]
        self.report["real_outliers"]["count"] = len(real_idxs)
        self.report["real_outliers"]["indices"] = real_idxs

    def _treat_outliers(self) -> None:
        """
        Drop or Winsorize “real outliers” based on model_family:
          – “linear” / “bayesian” → Winsorize  
          – else → drop rows
        """
        self._detect_univariate_outliers()
        self._detect_multivariate_outliers()
        self._filter_real_outliers()
        real_idxs = self.report["real_outliers"]["indices"]

        if len(real_idxs) == 0:
            if self.verbose:
                print("✔ No real outliers detected.")
            return

        if self.model_family in ["linear", "bayesian"] or getattr(self, "winsorize", False):
            if self.verbose:
                print(
                    f"⚠ Winsorizing {len(real_idxs)} rows (model_family={self.model_family})")
            for col in self.numeric_cols:
                arr = self.df[col]
                p1, p99 = np.percentile(arr.dropna(), [1, 99])
                self.df[col] = arr.clip(lower=p1, upper=p99)
        else:
            if self.verbose:
                print(
                    f"⚠ Dropping {len(real_idxs)} rows (model_family={self.model_family})")
            self.df.drop(index=real_idxs, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

    # ─────────────────── Numeric Transformation ────────────────────────────

    def _evaluate_transform(
        self, series: pd.Series
    ) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """
        For one numeric Series, test each method in transform_methods:
          – none, log1p, boxcox, yeo, quantile.
        Score by (Shapiro‐Wilk p-value, −|skew|). Return best_method and {method:(pval,skew)}.
        """
        data = series.dropna()
        scores: Dict[str, Tuple[float, float]] = {}
        if data.empty or data.nunique() < 5:
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

                # Shapiro–Wilk on up to 5000 samples
                samp = arr if arr.size <= 5000 else np.random.choice(
                    arr, 5000, replace=False)
                pval = stats.shapiro(samp)[1] if samp.size >= 3 else 0.0
                skew_abs = abs(stats.skew(arr))
                scores[method] = (pval, skew_abs)
            except Exception:
                continue

        # Choose by (pval, -skew_abs)
        best_method = "none"
        best_score = (-1.0, np.inf)
        for m, (pval, skew_abs) in scores.items():
            if (pval, -skew_abs) > best_score:
                best_score = (pval, -skew_abs)
                best_method = m

        return best_method, scores

    def _apply_transform(self) -> None:
        """
        Apply the chosen numeric transform to each numeric column. Record in report.
        """
        if self.verbose:
            print("\n🔍 Step 2: Numeric Transformation")

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
                # ensure positivity
                shifted = series + abs(min(series.dropna().min(), 0)) + 1
                self.df[col], _ = stats.boxcox(shifted.values)
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

    # ─────────────────── Numeric Imputation ──────────────────────────────

    def _random_sample_impute(self, orig: pd.Series) -> pd.Series:
        """Fill missing by sampling from non‐missing empirical distribution."""
        nonnull = orig.dropna().values
        full = orig.copy()
        mask = full.isna()
        if len(nonnull) == 0:
            return full.fillna(0)  # fallback if no non-missing
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
        For numeric col: return (ks_p, var_ratio, cov_change).
        """
        orig_nonnull = orig.dropna().values
        imp_nonnull = imputed.dropna().values
        ks_p = 0.0
        if orig_nonnull.size >= 2 and imp_nonnull.size >= 2:
            ks_p = stats.ks_2samp(orig_nonnull, imp_nonnull)[1]
        var_orig = np.nanvar(orig_nonnull)
        var_imp = np.nanvar(imp_nonnull)
        var_ratio = var_imp / var_orig if var_orig > 0 else np.nan

        idx = self.numeric_cols.index(col)
        diff = np.abs(cov_after[idx, :] - cov_before[idx, :])
        denom = np.abs(cov_before[idx, :])
        cov_change = np.sum(diff / (denom + 1e-9))

        return ks_p, var_ratio, cov_change

    def _impute_column_num(
        self, df_num: pd.DataFrame, col: str, cov_before: np.ndarray
    ) -> Tuple[str, pd.Series, Dict[str, Tuple[float, float, float]]]:
        """
        For numeric col, try each method in impute_methods_num,
        evaluate via KS/var_ratio/cov_change, choose best that meets thresholds.
        """
        orig = df_num[col]
        if orig.isna().sum() == 0:
            return "none", orig.copy(), {"none": (1.0, 1.0, 0.0)}

        results: Dict[str, Tuple[float, float, float]] = {}
        candidates: Dict[str, pd.Series] = {}

        for method in self.impute_methods_num:
            try:
                if method == "mean":
                    imp = SimpleImputer(strategy="mean")
                    arr = pd.Series(
                        imp.fit_transform(orig.values.reshape(-1, 1)).flatten(), index=orig.index
                    )
                elif method == "median":
                    imp = SimpleImputer(strategy="median")
                    arr = pd.Series(
                        imp.fit_transform(orig.values.reshape(-1, 1)).flatten(), index=orig.index
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
                        estimator=BayesianRidge(), sample_posterior=True,
                        random_state=0, max_iter=10
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

        # Select the best method that meets var_ratio ≥ 0.5 and cov_chg ≤ 0.2
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
            # Fallback to mean
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

    def _impute_missing_numeric(self) -> None:
        """
        Impute missing values in all numeric columns with QC.
        """
        if self.verbose:
            print("\n🔍 Step 4: Numeric Missing‐Value Imputation")

        df_num = self.df[self.numeric_cols].copy()
        complete_before = df_num.dropna()
        if complete_before.shape[0] < self.min_samples_per_feature:
            cov_before = np.zeros(
                (len(self.numeric_cols), len(self.numeric_cols)))
            if self.verbose:
                print("  • Skipping numeric covariance QC (too few complete cases).")
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
            method, arr, _ = self._impute_column_num(df_num, col, cov_before)
            self.df[col] = arr
            info = self.report["imputation"][col]
            if self.verbose:
                print(
                    f"  • {col:20s} → {info['chosen']}; "
                    f"KS_p={info['metrics']['ks_p']:.3f}, "
                    f"var_ratio={info['metrics']['var_ratio']:.3f}, "
                    f"cov_change={info['metrics']['cov_change']:.3f}"
                )

    # ─────────────────── Numeric Scaling ─────────────────────────────────

    def _select_scaler(self) -> None:
        """
        Choose a scaler for numeric columns based on model_family:
          – linear, bayesian, svm → StandardScaler  
          – knn → MinMaxScaler  
          – tree → None (trees prefer raw values)  
          – else → RobustScaler
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
        if scaler is not None and self.numeric_cols:
            self.df[self.numeric_cols] = scaler.fit_transform(
                self.df[self.numeric_cols])
            if self.verbose:
                print(f"🔍 Step 5: Applied {label} to numeric columns")

    # ────────────────── Numeric Dimensionality Reduction ─────────────────────

    def _validate_for_dr(self) -> Tuple[np.ndarray, str]:
        """
        Ensure DR is feasible. Check:
          1) No NaN in numeric portion  
          2) Enough samples: n_samples ≥ min_samples_per_feature × n_features  
          3) At least 2 numeric features  
          4) If LDA: target must exist and be categorical
        Returns (X_values, error_message). If error non‐empty, skip DR.
        """
        X = self.df[self.numeric_cols].copy()
        if X.isna().any().any():
            return X.values, "DR aborted: NaNs remain in numeric data."

        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < self.min_samples_per_feature * n_feats:
            return X.values, (
                f"DR aborted: insufficient samples ({n_samples}) for {n_feats} numeric features."
            )

        if self._dr_type == "lda":
            if not self.target or self.target not in self.df.columns:
                return X.values, "LDA aborted: no target column provided."
            if pd.api.types.is_numeric_dtype(self.df[self.target]):
                return X.values, "LDA aborted: target must be categorical for LDA."

        return X.values, ""

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
                f"\n🔍 DR: PCA → {n_comp} components (cumulative var={explained[n_comp-1]:.3f})")

    def _apply_kernel_pca(self, X: np.ndarray) -> None:
        gamma = 1.0 / X.shape[1]
        kpca = KernelPCA(kernel="rbf", gamma=gamma,
                         fit_inverse_transform=False)
        kpca.fit(X)
        self._dr_model = kpca
        self.report["dim_reduction"] = {
            "type": "KernelPCA",
            "params": {"kernel": "rbf", "gamma": gamma},
        }
        if self.verbose:
            print(f"\n🔍 DR: KernelPCA (rbf, gamma={gamma:.3f}) fitted")

    def _apply_tsvd(self, X: np.ndarray) -> None:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
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
                f"\n🔍 DR: TruncatedSVD → {chosen} components (cum var={explained[chosen-1]:.3f})")

    def _apply_lda(self, X: np.ndarray) -> None:
        y = self.df[self.target]
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        n_comp = min(len(np.unique(y)) - 1, X.shape[1])
        self._dr_model = lda
        self.report["dim_reduction"] = {
            "type": "LDA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
        }
        if self.verbose:
            print(
                f"\n🔍 DR: LDA → {n_comp} components (n_classes={len(np.unique(y))})")

    def _apply_dimensionality_reduction(self, X: np.ndarray) -> None:
        """
        Dispatch to the chosen DR method (PCA, KPCA, TSVD, LDA).
        """
        if self._dr_type == "pca":
            self._apply_pca(X)
        elif self._dr_type == "kpca":
            self._apply_kernel_pca(X)
        elif self._dr_type == "tsvd":
            self._apply_tsvd(X)
        elif self._dr_type == "lda":
            self._apply_lda(X)

    # ─────────────────── Categorical Missing Imputation ──────────────────────

    def _random_sample_impute_cat(self, orig: pd.Series) -> pd.Series:
        """
        Fill missing by sampling from non‐missing categories with equal probability.
        """
        nonnull = orig.dropna().values
        full = orig.copy().astype(object)
        mask = full.isna()
        if len(nonnull) == 0:
            return full.fillna("__MISSING__")
        full.loc[mask] = np.random.choice(
            nonnull, size=mask.sum(), replace=True)
        return full

    def _impute_column_cat(
        self, orig: pd.Series
    ) -> Tuple[str, pd.Series, Dict[str, float]]:
        """
        Try each method for a single categorical column:
          – mode: most frequent
          – constant_other: fill missing with "__MISSING__"
          – random_sample: sampling from empirical distribution
        Score each by simple frequency preservation:
          • Compare frequency distribution of non‐missing before vs after (e.g. via TVD or similar).
        Here we use “≥80% of modal frequency preserved” as a quick heuristic.
        """
        if orig.isna().sum() == 0:
            return "none", orig.copy(), {"none": 1.0}

        results: Dict[str, float] = {}
        candidates: Dict[str, pd.Series] = {}
        freq_before = orig.dropna().value_counts(normalize=True)

        for method in self.impute_methods_cat:
            try:
                if method == "mode":
                    imp = SimpleImputer(strategy="most_frequent")
                    arr = pd.Series(
                        imp.fit_transform(orig.values.reshape(-1, 1)).flatten(), index=orig.index
                    ).astype(object)
                elif method == "constant_other":
                    arr = orig.fillna("__MISSING__").astype(object)
                elif method == "random_sample":
                    arr = self._random_sample_impute_cat(orig)
                else:
                    continue

                freq_after = arr.value_counts(normalize=True)
                # Compute total variation distance on shared categories
                common = set(freq_before.index).intersection(
                    set(freq_after.index))
                tvd = sum(
                    abs(freq_before.loc[list(common)] - freq_after.loc[list(common)]))
                score = 1.0 - tvd  # higher is better
                results[method] = score
                candidates[method] = arr
            except Exception:
                continue

        # Choose method with max(score)
        if not results:
            # fallback to mode
            fallback = orig.fillna("__MISSING__").astype(object)
            self.report["categorical"][col] = {
                "missing_imputer": "fallback_constant",
                "score": 0.0
            }
            return "fallback_constant", fallback, {"fallback_constant": 0.0}

        best_method = max(results, key=results.get)
        best_score = results[best_method]
        best_arr = candidates[best_method]
        self.report["categorical"][col] = {
            "missing_imputer": best_method,
            "score": best_score
        }
        return best_method, best_arr, results

    def _impute_missing_categorical(self) -> None:
        """
        Impute missing values for all categorical columns and record decisions.
        """
        if self.verbose:
            print("\n🔍 Step 6: Categorical Missing‐Value Imputation")

        for col in self.categorical_cols:
            orig = self.df[col].astype(object)
            method, arr, _ = self._impute_column_cat(orig)
            self.df[col] = arr
            if self.verbose:
                print(f"  • {col:20s} → {method}")

    # ──────────────────── Rare‐Category Grouping ────────────────────────────

    def _group_rare_categories(self) -> None:
        """
        Any category whose relative frequency < self.rare_thresh is replaced with "__RARE__".
        Record the categories that were grouped under report["categorical"][col]["rare_grouped"].
        """
        if self.verbose:
            print("\n🔍 Step 7: Rare‐Category Grouping")

        for col in self.categorical_cols:
            freq = self.df[col].value_counts(normalize=True)
            rare_cats = set(freq[freq < self.rare_thresh].index)
            if not rare_cats:
                self.report["categorical"].setdefault(col, {})
                self.report["categorical"][col]["rare_grouped"] = []
                continue

            self.df[col] = self.df[col].apply(
                lambda x: "__RARE__" if x in rare_cats else x)
            self.report["categorical"][col]["rare_grouped"] = list(rare_cats)
            if self.verbose:
                print(
                    f"  • {col:20s} → grouped {len(rare_cats)} rare categories")

    # ────────────────────── Categorical Encoding ────────────────────────────

    def _target_encode_column(self, col: str) -> pd.Series:
        """
        Replace each category by the mean target value for that category.
        Requires that self.target is present and is non‐numeric or numeric categorical.
        """
        df = self.df[[col, self.target]].copy()
        # Compute category→mean(target)
        mapping = df.groupby(col)[self.target].mean()
        return self.df[col].map(mapping).fillna(mapping.mean())

    def _frequency_encode_column(self, col: str) -> pd.Series:
        """
        Replace each category by its frequency (normalized) in the column.
        """
        freq = self.df[col].value_counts(normalize=True)
        return self.df[col].map(freq).fillna(0.0)

    def _encode_column(self, col: str) -> Tuple[str, Union[pd.DataFrame, pd.Series]]:
        """
        Choose an encoding for a single categorical column based on model_family:
          – “linear” or “svm” → one‐hot (to maintain linear separability)  
          – “tree” → ordinal (trees can split ordered integers)  
          – “knn” → frequency (kNN uses numeric distances, freq may be smoother)  
          – “bayesian” → target encoding (Bayesian models can leverage numeric target means)  
        Returns (encoder_name, transformed_data). transformed_data is a DataFrame
        if one‐hot, or a Series otherwise.
        """
        col_data = self.df[col].astype(object)
        enc_choice = None
        transformed = None

        if self.model_family in ["linear", "svm"]:
            # One‐hot
            dummies = pd.get_dummies(
                col_data, prefix=col, drop_first=False, dtype=float)
            enc_choice = "onehot"
            transformed = dummies
        elif self.model_family == "tree":
            # Ordinal (label) but keep unseen as -1
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1)
            arr = enc.fit_transform(
                col_data.values.reshape(-1, 1)).astype(int).flatten()
            transformed = pd.Series(arr, name=col)
            enc_choice = "ordinal"
        elif self.model_family == "knn":
            # Frequency
            freq_series = self._frequency_encode_column(col)
            transformed = freq_series.rename(f"{col}_freq")
            enc_choice = "frequency"
        elif self.model_family == "bayesian" and self.target:
            # Target encoding
            te = self._target_encode_column(col)
            transformed = te.rename(f"{col}_target")
            enc_choice = "target"
        else:
            # Default fallback → frequency
            freq_series = self._frequency_encode_column(col)
            transformed = freq_series.rename(f"{col}_freq")
            enc_choice = "frequency"

        self.report["categorical"][col]["encoder"] = enc_choice
        return enc_choice, transformed

    def _encode_categorical(self) -> None:
        """
        For each categorical column:
          1) Impute missing (already done)
          2) Group rare categories (already done)
          3) Encode into numeric form
        Replace the original column with its encoded columns (one‐hot may expand columns).
        """
        if self.verbose:
            print("\n🔍 Step 8: Categorical Encoding")

        encoded_dfs = []
        for col in self.categorical_cols:
            enc_name, transformed = self._encode_column(col)
            if self.verbose:
                print(f"  • {col:20s} → {enc_name}")
            if isinstance(transformed, pd.Series):
                encoded_dfs.append(transformed.to_frame())
            else:
                # DataFrame for one‐hot
                encoded_dfs.append(transformed)

        # Drop original categorical columns, then concat encoded ones
        self.df.drop(columns=self.categorical_cols, inplace=True)
        if encoded_dfs:
            self.df = pd.concat(
                [self.df.reset_index(drop=True)] + encoded_dfs, axis=1)

    # ────────────────────── Main Pipeline Runner ────────────────────────────

    def run(self, dr_type: Optional[str] = "pca") -> pd.DataFrame:
        """
        Execute all steps in order:

          1) Numeric outlier detection & treatment  
          2) Numeric transformation  
          3) Numeric missing‐value imputation  
          4) Numeric scaling  
          5) Dimensionality reduction (if requested)  

          6) Categorical missing‐value imputation  
          7) Rare‐category grouping  
          8) Categorical encoding  

        dr_type: {"pca", "kpca", "tsvd", "lda", None}. If None, skip DR entirely.
        """
        # 1) Numeric outliers
        self._treat_outliers()

        # 2) Numeric transforms
        self._apply_transform()

        # 3) Numeric imputation
        self._impute_missing_numeric()

        # 4) Numeric scaling
        self._select_scaler()

        # 5) Dimensionality reduction
        if dr_type:
            self._dr_type = dr_type.lower()
            X_vals, error = self._validate_for_dr()
            if error:
                raise RuntimeError(error)
            self._apply_dimensionality_reduction(X_vals)

        # 6) Categorical imputation
        self._impute_missing_categorical()

        # 7) Rare‐category grouping
        self._group_rare_categories()

        # 8) Categorical encoding
        self._encode_categorical()

        return self.df.copy()

    def print_report(self) -> None:
        """
        Summarize all decisions:
          – Numeric outliers (uni + multi + real)  
          – Numeric transforms  
          – Numeric imputers + metrics  
          – Scaler used  
          – Dimensionality reduction details  
          – Categorical imputers + rare categories grouped + encoders  
          – List of non‐numeric columns (pre‐encoding)
        """
        print("\n" + "=" * 80)
        print("✅ DataQualityPipeline Report")
        print("=" * 80)

        # Univariate outliers
        print("\n▶ Univariate Outliers (per column, per method):")
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
            f"\n▶ Real Outliers (threshold={self.outlier_threshold} flags): count={ro['count']}")

        # Numeric transforms
        print("\n▶ Numeric Transformation Summary:")
        for col, info in self.report["transform"].items():
            print(f"  • {col:20s} → {info['chosen']}")

        # Numeric imputers
        print("\n▶ Numeric Imputation Summary:")
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
            if dtype in ["PCA", "TruncatedSVD"]:
                ev = dr["explained_variance_ratio"][nc - 1] if nc else None
                print(
                    f"  • {dtype}: n_orig={dr['n_original_features']}, n_comp={nc}, cum_var={ev:.3f}")
            elif dtype == "KernelPCA":
                print(f"  • KernelPCA: params={dr['params']}")
            elif dtype == "LDA":
                print(
                    f"  • LDA: n_orig={dr['n_original_features']}, n_comp={nc}")
        else:
            print("  • None")

        # Categorical
        print("\n▶ Categorical Processing:")
        for col, info in self.report["categorical"].items():
            imputer = info.get("missing_imputer", "none")
            rare = info.get("rare_grouped", [])
            encoder = info.get("encoder", "none")
            print(
                f"  • {col:20s} → imputer={imputer}, rare_grouped={len(rare)}, encoder={encoder}")

        # Non‐numeric columns (list before encoding)
        print("\n▶ Original Non‐numeric Columns (pre‐encoding):")
        if self.non_numeric_cols:
            print("  • " + ", ".join(self.non_numeric_cols))
        else:
            print("  • None")

        print("\n" + "=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (as a standalone script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataQualityPipeline Demo")
    parser.add_argument(
        "--input_csv", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--target", required=False, help="Target column (for LDA or target encoding)"
    )
    parser.add_argument(
        "--model_family",
        choices=["auto", "linear", "tree", "knn", "svm", "bayesian"],
        default="auto",
        help="Determine scaler/encoder/imputer preferences",
    )
    parser.add_argument(
        "--dr_type",
        choices=["pca", "kpca", "tsvd", "lda", "none"],
        default="pca",
        help="Which dimensionality reduction method to attempt (or none)",
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
