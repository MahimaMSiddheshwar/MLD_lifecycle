import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from scipy import stats
from scipy.stats import chi2, ks_2samp
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, OrdinalEncoder
)
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import EmpiricalCovariance

warnings.filterwarnings("ignore")


class DataQualityPipeline:
    """
    Comprehensive “dry‐run” and preprocessing pipeline:

      1) Numeric outlier detection & treatment
      2) Numeric transformation
      3) Numeric missing‐value imputation
      4) Numeric scaling
      5) Dimensionality reduction (PCA, TSVD, KPCA, or LDA)
      6) Categorical missing‐value imputation
      7) Rare‐category grouping
      8) Categorical encoding

    Usage:
        dq = DataQualityPipeline(df, target="target_column")
        df_processed = dq.run(dr_type="pca")
        dq.print_report()

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame including numeric & categorical features (and target).
    target : Optional[str]
        Name of the target column (needed for LDA or mutual‐info checks).
    outlier_threshold : int
        Minimum number of “votes” (from IQR, Z‐score, Modified Z, Mahalanobis)
        required to flag a row as a real outlier.
    """

    # Class-level constants
    UNIV_IQR_FACTOR = 1.5
    UNIV_ZSCORE_CUTOFF = 3.0
    UNIV_MODZ_CUTOFF = 3.5
    UNIV_FLAG_THRESHOLD = 2

    MULTI_CI = 0.975
    MIN_COMPLETE_FOR_COV = 5

    SKEW_THRESH_FOR_ROBUST = 1.0
    KURT_THRESH_FOR_ROBUST = 5.0
    SKEW_THRESH_FOR_STANDARD = 0.5

    TRANSFORM_CANDIDATES = ["none", "log1p", "yeo", "quantile"]

    RARE_FREQ_CUTOFF = 0.01

    ONEHOT_MAX_UNIQ = 0.05
    ORDINAL_MAX_UNIQ = 0.20
    KNOWNALLY_MAX_UNIQ = 0.50
    ULTRAHIGH_UNIQ = 0.50

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        outlier_threshold: int = 2,
        verbose: bool = True,
    ):
        self.raw = df.copy()
        self.df = df.copy()
        self.target = target
        self.outlier_threshold = outlier_threshold
        self.verbose = verbose

        # Identify columns
        self.numeric_cols: List[str] = self.df.select_dtypes(
            include=np.number).columns.tolist()
        if self.target and self.target in self.numeric_cols:
            self.numeric_cols.remove(self.target)
        self.categorical_cols: List[str] = [
            c for c in self.df.select_dtypes(include="object").columns.tolist()
            if c != self.target
        ]
        self.non_numeric_cols = self.categorical_cols.copy()

        # Internal models
        self.scaler_model = None
        self.dr_model = None

        # Report dictionary
        self.report: Dict[str, Dict] = {
            "univariate_outliers": {},
            "multivariate_outliers": {},
            "real_outliers": {},
            "transform": {},
            "imputation": {},
            "scaler": {},
            "dim_reduction": {},
            "categorical": {},
        }

        self._report_lines: List[str] = []

    # ────────────────────────────────────────────────────────────────────────────
    # 1) Numeric Outlier Detection & Treatment
    # ────────────────────────────────────────────────────────────────────────────

    def _iqr_outliers(self, series: pd.Series) -> List[int]:
        s = series.dropna()
        if s.empty:
            return []
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - self.UNIV_IQR_FACTOR * iqr, q3 + self.UNIV_IQR_FACTOR * iqr
        return s[(s < lb) | (s > ub)].index.tolist()

    def _zscore_outliers(self, series: pd.Series) -> List[int]:
        s = series.dropna()
        if len(s) < 2:
            return []
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            return []
        z = (s - mu) / sigma
        return s[np.abs(z) > self.UNIV_ZSCORE_CUTOFF].index.tolist()

    def _modz_outliers(self, series: pd.Series) -> List[int]:
        s = series.dropna()
        if len(s) < 2:
            return []
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        if mad == 0:
            return []
        modz = 0.6745 * (s - med) / mad
        return s[np.abs(modz) > self.UNIV_MODZ_CUTOFF].index.tolist()

    def _detect_univariate_outliers(self):
        self._report_lines.append("\nSTEP 1: Univariate Outlier Detection")
        votes: Dict[int, int] = {}
        for col in self.numeric_cols:
            self.report["univariate_outliers"][col] = {}
            s = self.df[col]
            # IQR
            try:
                iqr_idxs = self._iqr_outliers(s)
                cnt_iqr = len(iqr_idxs)
            except:
                iqr_idxs = []
                cnt_iqr = None
            self.report["univariate_outliers"][col]["iqr"] = cnt_iqr
            self._report_lines.append(
                f"  • {col:20s} | IQR flagged: {cnt_iqr}")
            for idx in iqr_idxs:
                votes[idx] = votes.get(idx, 0) + 1

            # Z-score
            try:
                z_idxs = self._zscore_outliers(s)
                cnt_z = len(z_idxs)
            except:
                z_idxs = []
                cnt_z = None
            self.report["univariate_outliers"][col]["zscore"] = cnt_z
            self._report_lines.append(
                f"  • {col:20s} | Z-score flagged: {cnt_z}")
            for idx in z_idxs:
                votes[idx] = votes.get(idx, 0) + 1

            # Modified Z
            try:
                modz_idxs = self._modz_outliers(s)
                cnt_modz = len(modz_idxs)
            except:
                modz_idxs = []
                cnt_modz = None
            self.report["univariate_outliers"][col]["modz"] = cnt_modz
            self._report_lines.append(
                f"  • {col:20s} | Modified Z flagged: {cnt_modz}")
            for idx in modz_idxs:
                votes[idx] = votes.get(idx, 0) + 1

        self._report_lines.append(
            "\nSTEP 2: Multivariate Outlier Detection (Mahalanobis)")
        numeric_df = self.df[self.numeric_cols].dropna()
        if numeric_df.shape[0] >= max(self.UNIV_FLAG_THRESHOLD * len(self.numeric_cols), self.MIN_COMPLETE_FOR_COV):
            cov = EmpiricalCovariance().fit(numeric_df.values)
            md = cov.mahalanobis(numeric_df.values)
            thresh = chi2.ppf(self.MULTI_CI, df=numeric_df.shape[1])
            multi_idxs = numeric_df.index[md > thresh].tolist()
            self.report["multivariate_outliers"]["mahalanobis"] = multi_idxs
            self._report_lines.append(
                f"  • Mahalanobis flagged: {len(multi_idxs)}")
            for idx in multi_idxs:
                votes[idx] = votes.get(idx, 0) + 1
        else:
            self.report["multivariate_outliers"]["mahalanobis"] = []
            self._report_lines.append("  • Not enough data for Mahalanobis")

        # Real outliers = those with votes ≥ threshold
        real_idxs = [idx for idx, v in votes.items() if v >=
                     self.outlier_threshold]
        self.report["real_outliers"] = {
            "indices": real_idxs, "count": len(real_idxs)}
        self._report_lines.append(
            f"\n  → Real outliers (votes ≥ {self.outlier_threshold}): {len(real_idxs)}")
        if real_idxs:
            self.df.drop(index=real_idxs, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self._report_lines.append(
                f"  • Dropped {len(real_idxs)} real outlier rows")

    # ────────────────────────────────────────────────────────────────────────────
    # 2) Numeric Transformation
    # ────────────────────────────────────────────────────────────────────────────

    def _evaluate_transform(self, series: pd.Series) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        s = series.dropna().values
        scores: Dict[str, Tuple[float, float]] = {}
        if len(s) < 5 or series.nunique() < 5:
            return "none", {"none": (1.0, 0.0)}

        for method in self.TRANSFORM_CANDIDATES:
            try:
                if method == "none":
                    arr = s
                elif method == "log1p":
                    if np.any(s <= -1):
                        continue
                    arr = np.log1p(s)
                elif method == "yeo":
                    pt = PowerTransformer(
                        method="yeo-johnson", standardize=True)
                    arr = pt.fit_transform(s.reshape(-1, 1)).flatten()
                else:  # quantile
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=0)
                    arr = qt.fit_transform(s.reshape(-1, 1)).flatten()
                sample = arr if len(arr) <= 5000 else np.random.choice(
                    arr, 5000, replace=False)
                if len(sample) >= 3:
                    try:
                        pval = float(stats.shapiro(sample)[1])
                    except:
                        pval = 0.0
                else:
                    pval = 0.0
                skew_abs = abs(float(stats.skew(arr)))
                scores[method] = (pval, skew_abs)
            except:
                continue

        best = "none"
        best_score = (-1.0, np.inf)
        for m, (pval, skew_abs) in scores.items():
            if (pval, -skew_abs) > best_score:
                best = m
                best_score = (pval, -skew_abs)

        return best, scores

    def _apply_transform(self):
        self._report_lines.append("\nSTEP 3: Numeric Transformation")
        for col in self.numeric_cols:
            orig = self.df[col]
            best, scores = self._evaluate_transform(orig)
            self.report["transform"][col] = {"chosen": best, "scores": scores}
            self._report_lines.append(f"  • {col:20s} → {best}, {scores}")
            if best == "log1p":
                self.df[col] = np.log1p(orig)
                self._report_lines.append(f"    • Applied log1p on '{col}'")
            elif best == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                self.df[col] = pt.fit_transform(
                    orig.values.reshape(-1, 1)).flatten()
                self._report_lines.append(
                    f"    • Applied Yeo-Johnson on '{col}'")
            elif best == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal", random_state=0)
                self.df[col] = qt.fit_transform(
                    orig.values.reshape(-1, 1)).flatten()
                self._report_lines.append(
                    f"    • Applied Quantile→Normal on '{col}'")
            else:
                self._report_lines.append(f"    • No transform for '{col}'")

    # ────────────────────────────────────────────────────────────────────────────
    # 3) Numeric Missing‐Value Imputation
    # ────────────────────────────────────────────────────────────────────────────

    def _random_sample_impute_num(self, series: pd.Series) -> pd.Series:
        s = series.copy()
        nonnull = s.dropna().values
        if len(nonnull) == 0:
            return s.fillna(0.0)
        mask = s.isna()
        s.loc[mask] = np.random.choice(nonnull, size=mask.sum(), replace=True)
        return s

    def _evaluate_impute_num(
        self, col: str, orig: pd.Series, imputed: pd.Series, cov_before: Optional[np.ndarray]
    ) -> Tuple[float, float, float]:
        orig_vals = orig.dropna().values
        imp_vals = imputed.dropna().values

        ks_p = 0.0
        if len(orig_vals) >= 2 and len(imp_vals) >= 2:
            try:
                ks_p = float(ks_2samp(orig_vals, imp_vals)[1])
            except:
                ks_p = 0.0

        var_orig = float(np.nanvar(orig_vals)) if len(
            orig_vals) > 0 else np.nan
        var_imp = float(np.nanvar(imp_vals)) if len(imp_vals) > 0 else np.nan
        var_ratio = var_imp / var_orig if var_orig and var_orig > 0 else np.nan

        if cov_before is None:
            cov_change = np.nan
        else:
            idx_feat = self.numeric_cols.index(col)
            temp = pd.DataFrame({c: self.df[c] for c in self.numeric_cols})
            temp[col] = imputed.values
            complete_idx = temp.dropna().index
            if len(complete_idx) < self.MIN_COMPLETE_FOR_COV:
                cov_change = np.nan
            else:
                cov_after = EmpiricalCovariance().fit(
                    temp.loc[complete_idx].values).covariance_
                diff = np.abs(cov_after[idx_feat, :] - cov_before[idx_feat, :])
                denom = np.abs(cov_before[idx_feat, :]) + 1e-9
                cov_change = float(np.sum(diff / denom))

        return ks_p, var_ratio, cov_change

    def _impute_missing_numeric(self):
        self._report_lines.append("\nSTEP 4: Numeric Missing-Value Imputation")
        df_num = self.df[self.numeric_cols].copy()
        complete_df = df_num.dropna()
        if complete_df.shape[0] < self.MIN_COMPLETE_FOR_COV:
            cov_before = None
            self._report_lines.append(
                "  • Too few complete rows → skipping covariance QC")
        else:
            cov_before = EmpiricalCovariance().fit(complete_df.values).covariance_

        for col in self.numeric_cols:
            orig = df_num[col]
            n_miss = int(orig.isna().sum())
            if n_miss == 0:
                self.report["imputation"][col] = {
                    "chosen": "none", "metrics": (1.0, 1.0, 0.0)}
                self._report_lines.append(f"  • {col:20s} → no missing")
                continue

            candidates: Dict[str, pd.Series] = {}
            metrics: Dict[str, Tuple[float, float, float]] = {}

            # mean
            try:
                imp = SimpleImputer(strategy="mean")
                arr = pd.Series(imp.fit_transform(
                    orig.values.reshape(-1, 1)).flatten(), index=orig.index)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["mean"] = arr
                metrics["mean"] = (ksp, vr, cc)
            except:
                pass

            # median
            try:
                imp = SimpleImputer(strategy="median")
                arr = pd.Series(imp.fit_transform(
                    orig.values.reshape(-1, 1)).flatten(), index=orig.index)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["median"] = arr
                metrics["median"] = (ksp, vr, cc)
            except:
                pass

            # knn
            try:
                imputer = KNNImputer(n_neighbors=5)
                tmp = df_num.copy()
                tmp_imp = pd.DataFrame(imputer.fit_transform(
                    tmp), columns=self.numeric_cols, index=df_num.index)
                arr = tmp_imp[col]
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["knn"] = arr
                metrics["knn"] = (ksp, vr, cc)
            except:
                pass

            # mice
            try:
                imputer = IterativeImputer(estimator=BayesianRidge(
                ), sample_posterior=True, random_state=0, max_iter=10)
                tmp = df_num.copy()
                tmp_imp = pd.DataFrame(imputer.fit_transform(
                    tmp), columns=self.numeric_cols, index=df_num.index)
                arr = tmp_imp[col]
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["mice"] = arr
                metrics["mice"] = (ksp, vr, cc)
            except:
                pass

            # random-sample
            try:
                arr = self._random_sample_impute_num(orig)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["random_sample"] = arr
                metrics["random_sample"] = (ksp, vr, cc)
            except:
                pass

            best_method = None
            best_score = (-1.0, -1.0, np.inf)
            for m, (ksp, vr, cc) in metrics.items():
                if (
                    ksp > best_score[0]
                    and vr >= 0.5
                    and (np.isnan(cc) or cc <= 0.2)
                ):
                    best_method = m
                    best_score = (ksp, vr, cc)

            if best_method is None:
                arr = orig.fillna(orig.mean())
                ksp_fb, vr_fb, _ = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                self.report["imputation"][col] = {
                    "chosen": "fallback_mean",
                    "metrics": (ksp_fb, vr_fb, np.nan),
                    "note": "none met QC"
                }
                self._report_lines.append(f"  • {col:20s} → fallback_mean")
                self.df[col] = arr
            else:
                self.report["imputation"][col] = {
                    "chosen": best_method,
                    "metrics": best_score,
                    "note": ""
                }
                self._report_lines.append(
                    f"  • {col:20s} → {best_method}, metrics={best_score}")
                self.df[col] = candidates[best_method]

    # ────────────────────────────────────────────────────────────────────────────
    # 4) Numeric Scaling
    # ────────────────────────────────────────────────────────────────────────────

    def _select_scaler(self):
        self._report_lines.append("\nSTEP 5: Numeric Scaling")
        if not self.numeric_cols:
            self.report["scaler"] = {
                "chosen": "none", "note": "no numeric columns"}
            self._report_lines.append("  • No numeric cols to scale")
            return

        skews = {
            c: float(stats.skew(self.df[c].dropna().values)) for c in self.numeric_cols}
        kurts = {c: float(stats.kurtosis(
            self.df[c].dropna().values)) for c in self.numeric_cols}

        if any(abs(sk) > self.SKEW_THRESH_FOR_ROBUST or abs(ku) > self.KURT_THRESH_FOR_ROBUST
               for sk, ku in zip(skews.values(), kurts.values())):
            scaler_name = "RobustScaler"
            model = RobustScaler()
        elif all(abs(sk) < self.SKEW_THRESH_FOR_STANDARD for sk in skews.values()):
            scaler_name = "StandardScaler"
            model = StandardScaler()
        else:
            scaler_name = "MinMaxScaler"
            model = MinMaxScaler()

        self.report["scaler"] = {"chosen": scaler_name,
                                 "skew": skews, "kurtosis": kurts}
        self._report_lines.append(f"  • Chosen scaler: {scaler_name}")
        self.df[self.numeric_cols] = model.fit_transform(
            self.df[self.numeric_cols])
        self.scaler_model = model
        self._report_lines.append(f"  • Applied {scaler_name}")

    # ────────────────────────────────────────────────────────────────────────────
    # 5) Dimensionality Reduction
    # ────────────────────────────────────────────────────────────────────────────

    def _validate_for_dr(self) -> Tuple[np.ndarray, Optional[str]]:
        X = self.df[self.numeric_cols].copy()
        if X.isna().any(axis=None):
            return X.values, "PCA aborted: NaNs present"
        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < 5 * n_feats:
            return X.values, f"PCA aborted: too few (n={n_samples}, p={n_feats})"
        return X.values, None

    def _apply_dimensionality_reduction(self, X_vals: np.ndarray):
        method = self._dr_type.lower()
        n_samples, n_feats = X_vals.shape
        if method == "pca":
            pca = PCA().fit(X_vals)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cumvar, 0.9) + 1)
            self.dr_model = PCA(n_components=n_comp)
            X_red = self.dr_model.fit_transform(X_vals)
            self.report["dim_reduction"] = {
                "type": "PCA",
                "n_original_features": n_feats,
                "n_components": n_comp,
                "explained_variance_ratio": cumvar.tolist(),
            }
        elif method == "tsvd":
            tsvd = TruncatedSVD(n_components=min(n_feats, 10), random_state=0)
            X_red = tsvd.fit_transform(X_vals)
            self.dr_model = tsvd
            self.report["dim_reduction"] = {
                "type": "TruncatedSVD",
                "n_original_features": n_feats,
                "n_components": tsvd.n_components,
                "explained_variance_ratio": tsvd.explained_variance_ratio_.tolist(),
            }
        elif method == "kpca":
            kp = KernelPCA(n_components=min(n_feats, 10),
                           kernel="rbf", random_state=0)
            X_red = kp.fit_transform(X_vals)
            self.dr_model = kp
            self.report["dim_reduction"] = {
                "type": "KernelPCA",
                "n_original_features": n_feats,
                "n_components": kp.n_components,
                "kernel": "rbf",
            }
        elif method == "lda":
            if not self.target or self.target not in self.df.columns:
                raise RuntimeError(
                    "LDA requires a target column present in DataFrame")
            y = self.df[self.target]
            if y.nunique() < 2:
                raise RuntimeError(
                    "LDA requires at least two distinct classes")
            n_comp = min(y.nunique() - 1, n_feats)
            lda = LinearDiscriminantAnalysis(n_components=n_comp)
            X_red = lda.fit_transform(X_vals, y.loc[~np.isnan(y.values)])
            self.dr_model = lda
            self.report["dim_reduction"] = {
                "type": "LDA",
                "n_original_features": n_feats,
                "n_components": n_comp,
            }
        else:
            raise RuntimeError(f"Unknown DR method: {self._dr_type}")

        cols = [f"DR{i+1}" for i in range(X_red.shape[1])]
        df_red = pd.DataFrame(X_red, columns=cols, index=self.df.index)
        self.df = pd.concat(
            [self.df.drop(columns=self.numeric_cols), df_red], axis=1)
        self._report_lines.append(
            f"  • Applied {self.report['dim_reduction']['type']}")

    # ────────────────────────────────────────────────────────────────────────────
    # 6) Categorical Missing‐Value Imputation
    # ────────────────────────────────────────────────────────────────────────────

    def _random_sample_impute_cat(self, series: pd.Series) -> pd.Series:
        s = series.astype(object).copy()
        nonnull = s.dropna().values
        if len(nonnull) == 0:
            return s.fillna("__MISSING__")
        mask = s.isna()
        s.loc[mask] = np.random.choice(nonnull, size=mask.sum(), replace=True)
        return s

    def _evaluate_impute_cat(self, series: pd.Series) -> Tuple[str, Dict[str, float]]:
        orig = series.astype(object)
        freq_before = orig.dropna().value_counts(normalize=True)
        candidates: Dict[str, pd.Series] = {}
        scores: Dict[str, float] = {}

        # mode
        try:
            imp = SimpleImputer(strategy="most_frequent")
            arr = pd.Series(imp.fit_transform(
                orig.values.reshape(-1, 1)).flatten(), index=orig.index).astype(object)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            candidates["mode"] = arr
            scores["mode"] = 1.0 - tvd
        except:
            pass

        # constant_other
        try:
            arr = orig.fillna("__MISSING__").astype(object)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            candidates["constant_other"] = arr
            scores["constant_other"] = 1.0 - tvd
        except:
            pass

        # random_sample
        try:
            arr = self._random_sample_impute_cat(orig)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            candidates["random_sample"] = arr
            scores["random_sample"] = 1.0 - tvd
        except:
            pass

        if not scores:
            return "fallback_constant", {"fallback_constant": 0.0}
        best = max(scores, key=scores.get)
        return best, scores

    def _impute_missing_categorical(self):
        self._report_lines.append(
            "\nSTEP 6: Categorical Missing-Value Imputation")
        for col in self.categorical_cols:
            orig = self.df[col].astype(object)
            n_miss = int(orig.isna().sum())
            if n_miss == 0:
                self.report["categorical"][col] = {
                    "missing_imputer": "none", "rare_grouped": [], "encoder": "none"
                }
                self._report_lines.append(f"  • {col:20s} → no missing")
                continue

            best, scores = self._evaluate_impute_cat(orig)
            self.report["categorical"][col] = {
                "missing_imputer": best, "rare_grouped": [], "encoder": "none"}
            self._report_lines.append(
                f"  • {col:20s} → {best}, scores={scores}")
            if best == "mode":
                imp = SimpleImputer(strategy="most_frequent")
                self.df[col] = pd.Series(imp.fit_transform(
                    orig.values.reshape(-1, 1)).flatten(), index=orig.index).astype(object)
                self._report_lines.append(f"    • Mode imputed '{col}'")
            elif best == "constant_other":
                self.df[col] = orig.fillna("__MISSING__").astype(object)
                self._report_lines.append(
                    f"    • Filled missing with '__MISSING__' in '{col}'")
            elif best == "random_sample":
                self.df[col] = self._random_sample_impute_cat(orig)
                self._report_lines.append(
                    f"    • Random-sample imputed '{col}'")
            else:
                self.df[col] = orig.fillna("__MISSING__").astype(object)
                self._report_lines.append(
                    f"    • Fallback filled '__MISSING__' in '{col}'")

    # ────────────────────────────────────────────────────────────────────────────
    # 7) Rare‐Category Grouping
    # ────────────────────────────────────────────────────────────────────────────

    def _group_rare_categories(self):
        self._report_lines.append("\nSTEP 7: Rare-Category Grouping")
        for col in self.categorical_cols:
            freq = self.df[col].value_counts(normalize=True)
            rare = set(freq[freq < self.RARE_FREQ_CUTOFF].index)
            if not rare:
                self.report["categorical"][col]["rare_grouped"] = []
                self._report_lines.append(f"  • {col:20s} → no rare")
                continue
            self.report["categorical"][col]["rare_grouped"] = list(rare)
            self._report_lines.append(
                f"  • {col:20s} → grouped {len(rare)} rare")
            self.df[col] = self.df[col].apply(
                lambda x: "__RARE__" if x in rare else x)

    # ────────────────────────────────────────────────────────────────────────────
    # 8) Categorical Encoding
    # ────────────────────────────────────────────────────────────────────────────

    def _onehot_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return pd.get_dummies(df[cols], prefix=cols, drop_first=False, dtype=float)

    def _ordinal_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)
        arr = enc.fit_transform(df[cols].astype(object))
        return pd.DataFrame(arr.astype(int), columns=cols, index=df.index)

    def _frequency_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out: Dict[str, List[float]] = {}
        for col in cols:
            freq = df[col].value_counts(normalize=True)
            out[col + "_freq"] = df[col].map(freq).fillna(0.0)
        return pd.DataFrame(out, index=df.index)

    def _encode_categorical(self):
        self._report_lines.append("\nSTEP 8: Categorical Encoding")
        base = self.df.copy()
        n_rows = len(base)

        # Determine per-column encoding based on cardinality
        col_info = {}
        for col in self.categorical_cols:
            uniq = base[col].nunique()
            frac = uniq / n_rows if n_rows > 0 else 0.0
            col_info[col] = {"uniq": uniq, "frac": frac}

        # 8a) One-hot if frac ≤ ONEHOT_MAX_UNIQ
        onehot_cols = [c for c, info in col_info.items(
        ) if info["frac"] <= self.ONEHOT_MAX_UNIQ]
        # 8b) Ordinal if ONEHOT_MAX_UNIQ < frac ≤ ORDINAL_MAX_UNIQ
        ordinal_cols = [c for c, info in col_info.items()
                        if self.ONEHOT_MAX_UNIQ < info["frac"] <= self.ORDINAL_MAX_UNIQ]
        # 8c) Frequency if ORDINAL_MAX_UNIQ < frac ≤ KNOWNALLY_MAX_UNIQ
        freq_cols = [c for c, info in col_info.items()
                     if self.ORDINAL_MAX_UNIQ < info["frac"] <= self.KNOWNALLY_MAX_UNIQ]
        # 8d) Suggest target encoding if frac > ULTRAHIGH_UNIQ
        suggest_cols = [c for c, info in col_info.items(
        ) if info["frac"] > self.ULTRAHIGH_UNIQ]

        # Apply one-hot
        if onehot_cols:
            oh = self._onehot_encode(base, onehot_cols)
            self.df = pd.concat(
                [self.df.drop(columns=onehot_cols), oh], axis=1)
            for c in onehot_cols:
                self.report["categorical"][c]["encoder"] = "onehot"
            self._report_lines.append(f"  • One-hot encoded: {onehot_cols}")

        # Apply ordinal
        if ordinal_cols:
            ord_df = self._ordinal_encode(self.df, ordinal_cols)
            self.df = pd.concat(
                [self.df.drop(columns=ordinal_cols), ord_df], axis=1)
            for c in ordinal_cols:
                self.report["categorical"][c]["encoder"] = "ordinal"
            self._report_lines.append(f"  • Ordinal encoded: {ordinal_cols}")

        # Apply frequency
        if freq_cols:
            freq_df = self._frequency_encode(self.df, freq_cols)
            self.df = pd.concat(
                [self.df.drop(columns=freq_cols), freq_df], axis=1)
            for c in freq_cols:
                self.report["categorical"][c]["encoder"] = "frequency"
            self._report_lines.append(f"  • Frequency encoded: {freq_cols}")

        # Record suggestions
        for c in suggest_cols:
            self.report["categorical"][c]["encoder"] = "suggest_target"
            self._report_lines.append(f"  • Suggest target encoding for: {c}")

    # ────────────────────────────────────────────────────────────────────────────
    # 9) Main Pipeline Runner
    # ────────────────────────────────────────────────────────────────────────────

    def run(self, dr_type: Optional[str] = "pca") -> pd.DataFrame:
        """
        Execute steps in order:

          1) Numeric outlier detection & treatment  
          2) Numeric transformation  
          3) Numeric missing‐value imputation  
          4) Numeric scaling  
          5) Dimensionality reduction (if requested)  

          6) Categorical missing‐value imputation  
          7) Rare‐category grouping  
          8) Categorical encoding  

        dr_type: {"pca", "kpca", "tsvd", "lda", None}
        """
        # 1) Numeric outliers
        self._detect_univariate_outliers()

        # 2) Numeric transforms
        self._apply_transform()

        # 3) Numeric imputation
        self._impute_missing_numeric()

        # 4) Numeric scaling
        self._select_scaler()

        # 5) Dimensionality reduction
        if dr_type:
            self._dr_type = dr_type
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

    # ────────────────────────────────────────────────────────────────────────────
    # Print Report
    # ────────────────────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        print("\n" + "=" * 80)
        print("✅ DataQualityPipeline Report")
        print("=" * 80)

        # Univariate outliers
        print("\n▶ Univariate Outliers (per column, per method):")
        for col, methods_dict in self.report["univariate_outliers"].items():
            if methods_dict:
                line = "  • " + col + ": " + ", ".join(
                    f"{m}={count}" for m, count in methods_dict.items() if count is not None
                )
                print(line)
            else:
                print(f"  • {col}: None")

        # Multivariate outliers
        print("\n▶ Multivariate Outliers:")
        for method, idxs in self.report["multivariate_outliers"].items():
            print(f"  • {method}: count={len(idxs)}")

        # Real outliers
        ro = self.report["real_outliers"]
        print(
            f"\n▶ Real Outliers (threshold={self.outlier_threshold} votes): count={ro['count']}")

        # Numeric transforms
        print("\n▶ Numeric Transformation Summary:")
        for col, info in self.report["transform"].items():
            print(f"  • {col:20s} → {info['chosen']}")

        # Numeric imputers
        print("\n▶ Numeric Imputation Summary:")
        for col, info in self.report["imputation"].items():
            ch = info["chosen"]
            ksp, vr, cc = info["metrics"]
            print(
                f"  • {col:20s} → {ch}, KS_p={ksp:.3f}, var_ratio={vr:.3f}, cov_change={cc:.3f}")

        # Scaler
        sc = self.report["scaler"].get("chosen", "none")
        print(f"\n▶ Scaler: {sc}")

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
                print(f"  • KernelPCA: params: n_comp={dr['n_components']}")
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

        # Non‐numeric columns (pre-encoding)
        print("\n▶ Original Non‐numeric Columns (pre‐encoding):")
        if self.non_numeric_cols:
            print("  • " + ", ".join(self.non_numeric_cols))
        else:
            print("  • None")

        print("\n" + "=" * 80 + "\n")
