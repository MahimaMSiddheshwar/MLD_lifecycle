#!/usr/bin/env python3
"""
MLDLC End-to-End Pipeline
=========================

This script defines a single class `MLProjectPipeline` that orchestrates all
phases of a classical machine‐learning lifecycle up through model‐ready data:
  1) Data Quality (duplicates, missing, outliers, scaling, transformation, PCA)
  2) Exploratory Data Analysis (basic + advanced)
  3) Probabilistic Analysis (distributions, entropy, MI, copula, etc.)
  4) Feature Selection (nzv, correlation, mutual information/F-score)
  5) Feature Engineering (scaling, encoding, rare grouping, etc.)
  6) Split & Baseline Benchmark (train/val/test, baseline metrics, sanity checks)

After running, it will produce:
  • `analysis_report.txt` with detailed decisions & suggestions
  • `reports/eda/…` (CSV summaries, plots) for EDA
  • `reports/prob/…` (distribution fits, entropy tables) for probabilistic analysis
  • `reports/feature/feature_audit.json` + `reports/feature/feature_shape.txt`
  • `data/splits/{train,val,test}.parquet` + `reports/baseline/baseline_metrics.json`
  • `models/preprocessor.joblib`, `models/preprocessor_manifest.json`

Usage (from project root):
    python3 ml_pipeline.py --input data/raw/your_data.csv --target is_churn

Ensure the following packages are installed:
    numpy, pandas, scipy, scikit-learn, imbalanced-learn, category_encoders,
    ydata-profiling, matplotlib, seaborn, copulas
"""

import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Callable, Union
from statsmodels.imputation.mice import mice

import scipy.stats as stats
from scipy.stats import chi2, ks_2samp, entropy as scipy_entropy

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, OrdinalEncoder
)
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif,
    mutual_info_regression, f_classif
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE
from category_encoders import TargetEncoder, HashingEncoder, WOEEncoder

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from copulas.multivariate import GaussianMultivariate
from ydata_profiling import ProfileReport

warnings.filterwarnings("ignore")


# =============================================================================
# 1) DataQualityPipeline
# =============================================================================

class DataQualityPipeline(BaseEstimator, TransformerMixin):
    """
    Phase 3: Data Quality & Preprocessing

      1. Drop duplicates
      2. Numeric missing-value imputation (mean, median, KNN, MICE, random-sample)
      3. Univariate & Multivariate outlier detection & removal
      4. Scaling (Standard, MinMax, or Robust based on skew/kurtosis)
      5. Extra transformation (log1p, Yeo-Johnson, Quantile→Normal if needed)
      6. Categorical missing-value imputation (mode, constant, random-sample)
      7. Rare-category grouping (<1% → "__RARE__")
      8. Categorical encoding variants (linear, tree, knn) → writes 3 CSVs
      9. PCA on numeric → replaces numeric columns with PCs

    Produces:
      - analysis_report.txt
      - processed_linear.csv
      - processed_tree.csv
      - processed_knn.csv

    Class-level thresholds (adjust as needed):
      MIN_COMPLETE_FOR_COV = 5
      VARIANCE_RATIO_CUTOFF = 0.50
      COV_CHANGE_CUTOFF = 0.20
      UNIV_IQR_FACTOR = 1.5
      UNIV_ZSCORE_CUTOFF = 3.0
      UNIV_MODZ_CUTOFF = 3.5
      UNIV_FLAG_THRESHOLD = 2
      MULTI_CI = 0.975
      SKEW_THRESH_FOR_ROBUST = 1.0
      KURT_THRESH_FOR_ROBUST = 5.0
      SKEW_THRESH_FOR_STANDARD = 0.5
      TRANSFORM_CANDIDATES = ["none", "log1p", "yeo", "quantile"]
      CAT_IMPUTE_TVD_CUTOFF = 0.2
      RARE_FREQ_CUTOFF = 0.01
      ONEHOT_MAX_UNIQ = 0.05
      ORDINAL_MAX_UNIQ = 0.20
      KNOWNALLY_MAX_UNIQ = 0.50
      ULTRAHIGH_UNIQ = 0.50
    """

    # Class-level constants
    MIN_COMPLETE_FOR_COV = 5
    VARIANCE_RATIO_CUTOFF = 0.50
    COV_CHANGE_CUTOFF = 0.20

    UNIV_IQR_FACTOR = 1.5
    UNIV_ZSCORE_CUTOFF = 3.0
    UNIV_MODZ_CUTOFF = 3.5
    UNIV_FLAG_THRESHOLD = 2

    MULTI_CI = 0.975

    SKEW_THRESH_FOR_ROBUST = 1.0
    KURT_THRESH_FOR_ROBUST = 5.0
    SKEW_THRESH_FOR_STANDARD = 0.5

    TRANSFORM_CANDIDATES = ["none", "log1p", "yeo", "quantile"]

    CAT_IMPUTE_TVD_CUTOFF = 0.2
    RARE_FREQ_CUTOFF = 0.01

    ONEHOT_MAX_UNIQ = 0.05
    ORDINAL_MAX_UNIQ = 0.20
    KNOWNALLY_MAX_UNIQ = 0.50
    ULTRAHIGH_UNIQ = 0.50

    def __init__(
        self,
        target_column: Optional[str] = None,
        pca_variance_threshold: float = 0.90,
        verbose: bool = True,
    ):
        self.target = target_column
        self.pca_variance_threshold = pca_variance_threshold
        self.verbose = verbose

        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.non_numeric_cols: List[str] = []

        self.scaler_model = None
        self.pca_model = None

        self.report: Dict[str, Dict] = {
            "duplicates": {},
            "missing": {},
            "univariate_outliers": {},
            "multivariate_outliers": {},
            "real_outliers": {},
            "scaler": {},
            "transform": {},
            "categorical_imputation": {},
            "rare_categories": {},
            "encoding": {},
            "pca": {},
        }

        self._report_lines: List[str] = []
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------
    # 1) Drop duplicates
    # ------------------------------------------------------------
    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        dup_mask = df.duplicated(keep="first")
        n_dups = int(dup_mask.sum())
        self.report["duplicates"] = {"count": n_dups}
        self._report_lines.append(f"STEP 1: Found {n_dups} duplicate rows.")
        if n_dups > 0:
            df2 = df.loc[~dup_mask].reset_index(drop=True)
            self._report_lines.append(f"  • Dropped {n_dups} duplicates.")
            return df2
        return df

    # ------------------------------------------------------------
    # 2) Numeric missing-value imputation
    # ------------------------------------------------------------
    def _random_sample_impute_num(self, orig: pd.Series) -> pd.Series:
        nonnull = orig.dropna().values
        out = orig.copy()
        mask = out.isna()
        if len(nonnull) == 0:
            return out.fillna(0.0)
        out.loc[mask] = np.random.choice(
            nonnull, size=mask.sum(), replace=True)
        return out

    def _evaluate_impute_num(
        self, col: str, orig: pd.Series, imputed: pd.Series, cov_before: Optional[np.ndarray]
    ) -> Tuple[float, float, float]:
        orig_nonnull = orig.dropna().values
        imp_nonnull = imputed.dropna().values

        ks_p = 0.0
        if len(orig_nonnull) >= 2 and len(imp_nonnull) >= 2:
            try:
                ks_p = float(ks_2samp(orig_nonnull, imp_nonnull)[1])
            except:
                ks_p = 0.0

        var_orig = float(np.nanvar(orig_nonnull)) if len(
            orig_nonnull) > 0 else np.nan
        var_imp = float(np.nanvar(imp_nonnull)) if len(
            imp_nonnull) > 0 else np.nan
        var_ratio = var_imp / var_orig if var_orig and var_orig > 0 else np.nan

        if cov_before is None:
            cov_change = np.nan
        else:
            idx_feat = self.numeric_cols.index(col)
            temp = pd.DataFrame({c: self._df[c] for c in self.numeric_cols})
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

    def _impute_missing_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = df[self.numeric_cols].copy()
        complete_df = df_num.dropna()
        if complete_df.shape[0] < self.MIN_COMPLETE_FOR_COV:
            cov_before = None
            self._report_lines.append(
                "STEP 2: Too few complete rows → skipping covariance QC.")
        else:
            cov_before = EmpiricalCovariance().fit(complete_df.values).covariance_

        for col in self.numeric_cols:
            orig = df_num[col]
            n_missing = int(orig.isna().sum())
            if n_missing == 0:
                self.report["missing"][col] = {
                    "chosen": "none", "note": "no missing"}
                self._report_lines.append(f"  • {col:20s} → no missing")
                continue

            candidates: Dict[str, pd.Series] = {}
            scores: Dict[str, Tuple[float, float, float]] = {}

            # mean
            try:
                imp = SimpleImputer(strategy="mean")
                arr = pd.Series(imp.fit_transform(
                    orig.values.reshape(-1, 1)).flatten(), index=orig.index)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["mean"] = arr
                scores["mean"] = (ksp, vr, cc)
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
                scores["median"] = (ksp, vr, cc)
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
                scores["knn"] = (ksp, vr, cc)
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
                scores["mice"] = (ksp, vr, cc)
            except:
                pass

            # random_sample
            try:
                arr = self._random_sample_impute_num(orig)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["random_sample"] = arr
                scores["random_sample"] = (ksp, vr, cc)
            except:
                pass

            best_method = None
            best_score = (-1.0, -1.0, np.inf)
            for m, (ksp, vr, cc) in scores.items():
                if (ksp > best_score[0]
                    and vr >= self.VARIANCE_RATIO_CUTOFF
                        and (np.isnan(cc) or cc <= self.COV_CHANGE_CUTOFF)):
                    best_method = m
                    best_score = (ksp, vr, cc)

            if best_method is None:
                arr = orig.fillna(orig.mean())
                ksp_fb, vr_fb, _ = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                self.report["missing"][col] = {
                    "chosen": "fallback_mean",
                    "metrics": (ksp_fb, vr_fb, np.nan),
                    "note": "none met QC",
                }
                self._report_lines.append(f"  • {col:20s} → fallback_mean")
                df[col] = arr
            else:
                self.report["missing"][col] = {
                    "chosen": best_method,
                    "metrics": best_score,
                    "note": "",
                }
                self._report_lines.append(
                    f"  • {col:20s} → {best_method}, metrics={best_score}")
                df[col] = candidates[best_method]

        return df

    # ------------------------------------------------------------
    # 3) Outlier detection & treatment (numeric)
    # ------------------------------------------------------------
    def _iqr_outliers(self, s: pd.Series) -> List[int]:
        arr = s.dropna().values
        if len(arr) == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - self.UNIV_IQR_FACTOR * iqr, q3 + self.UNIV_IQR_FACTOR * iqr
        return s[(s < lb) | (s > ub)].index.tolist()

    def _zscore_outliers(self, s: pd.Series) -> List[int]:
        arr = s.dropna().values
        if len(arr) < 2:
            return []
        mu, sigma = s.mean(), s.std()
        if sigma == 0:
            return []
        z = (arr - mu) / sigma
        return s[np.abs(z) > self.UNIV_ZSCORE_CUTOFF].index.tolist()

    def _modz_outliers(self, s: pd.Series) -> List[int]:
        arr = s.dropna().values
        if len(arr) < 2:
            return []
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0:
            return []
        modz = 0.6745 * (arr - med) / mad
        return s[np.abs(modz) > self.UNIV_MODZ_CUTOFF].index.tolist()

    def _detect_outliers(self):
        self._report_lines.append("\nSTEP 3: Outlier Detection & Treatment")
        univ_votes: Dict[int, int] = {}
        for col in self.numeric_cols:
            self.report["univariate_outliers"][col] = {}
            s = self._df[col]
            for method in ["iqr", "zscore", "modz"]:
                try:
                    if method == "iqr":
                        idxs = self._iqr_outliers(s)
                    elif method == "zscore":
                        idxs = self._zscore_outliers(s)
                    else:
                        idxs = self._modz_outliers(s)
                    cnt = len(idxs)
                    self.report["univariate_outliers"][col][method] = cnt
                    self._report_lines.append(
                        f"  • {col:20s} via {method:>6s} → {cnt} flagged")
                    for i in idxs:
                        univ_votes[i] = univ_votes.get(i, 0) + 1
                except:
                    self.report["univariate_outliers"][col][method] = None
                    self._report_lines.append(
                        f"  • {col:20s} via {method:>6s} → error")

        self._report_lines.append(
            "\nSTEP 4: Multivariate Outlier Detection (Mahalanobis)")
        numeric_only = self._df[self.numeric_cols].dropna()
        if numeric_only.shape[0] >= max(self.UNIV_FLAG_THRESHOLD * len(self.numeric_cols), 5):
            cov = EmpiricalCovariance().fit(numeric_only.values)
            md = cov.mahalanobis(numeric_only.values)
            thresh = chi2.ppf(self.MULTI_CI, df=numeric_only.shape[1])
            mask = md > thresh
            idxs = numeric_only.index[mask]
            self.report["multivariate_outliers"]["mahalanobis"] = list(idxs)
            self._report_lines.append(f"  • mahalanobis → {len(idxs)} flagged")
            for i in idxs:
                univ_votes[i] = univ_votes.get(i, 0) + 1
        else:
            self.report["multivariate_outliers"]["mahalanobis"] = []
            self._report_lines.append(
                "  • Not enough complete rows → skipped Mahalanobis")

        real = [idx for idx, votes in univ_votes.items() if votes >=
                self.UNIV_FLAG_THRESHOLD]
        self.report["real_outliers"] = {"indices": real, "count": len(real)}
        self._report_lines.append(
            f"  → Real outliers (≥{self.UNIV_FLAG_THRESHOLD} votes): {len(real)}")
        if real:
            self._df.drop(index=real, inplace=True)
            self._df.reset_index(drop=True, inplace=True)
            self._report_lines.append(
                f"  • Dropped {len(real)} real outlier rows.")

    # ------------------------------------------------------------
    # 4) Scaling (numeric)
    # ------------------------------------------------------------
    def _choose_scaler(self) -> Tuple[str, Dict[str, Dict[str, float]]]:
        skews, kurts = {}, {}
        for col in self.numeric_cols:
            arr = self._df[col].dropna().values
            if len(arr) > 2:
                skews[col] = float(stats.skew(arr))
                kurts[col] = float(stats.kurtosis(arr))
            else:
                skews[col] = 0.0
                kurts[col] = 0.0

        if any(abs(sk) > self.SKEW_THRESH_FOR_ROBUST or abs(ku) > self.KURT_THRESH_FOR_ROBUST
               for sk, ku in zip(skews.values(), kurts.values())):
            return "RobustScaler", {"skew": skews, "kurtosis": kurts}

        if all(abs(sk) < self.SKEW_THRESH_FOR_STANDARD for sk in skews.values()):
            return "StandardScaler", {"skew": skews, "kurtosis": kurts}

        return "MinMaxScaler", {"skew": skews, "kurtosis": kurts}

    def _apply_scaling(self):
        self._report_lines.append("\nSTEP 5: Scaling (Numeric)")
        if not self.numeric_cols:
            self.report["scaler"] = {
                "chosen": "none", "note": "no numeric columns"}
            self._report_lines.append("  • No numeric columns to scale.")
            return

        scaler_name, metrics = self._choose_scaler()
        self.report["scaler"] = {"chosen": scaler_name, "metrics": metrics}
        self._report_lines.append(
            f"  • Suggested scaler: {scaler_name}, stats={metrics}")

        if scaler_name == "StandardScaler":
            model = StandardScaler()
        elif scaler_name == "MinMaxScaler":
            model = MinMaxScaler()
        else:
            model = RobustScaler()

        self._df[self.numeric_cols] = model.fit_transform(
            self._df[self.numeric_cols])
        self.scaler_model = model
        self._report_lines.append(f"  • Applied {scaler_name}.")

    # ------------------------------------------------------------
    # 5) Extra transformation (numeric)
    # ------------------------------------------------------------
    def _evaluate_transform(self, series: pd.Series) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        arr_full = series.dropna().values
        scores: Dict[str, Tuple[float, float]] = {}
        if len(arr_full) < 5 or series.nunique() < 5:
            return "none", {"none": (1.0, 0.0)}

        for method in self.TRANSFORM_CANDIDATES:
            try:
                if method == "none":
                    arr = arr_full
                elif method == "log1p":
                    if np.any(arr_full <= -1):
                        continue
                    arr = np.log1p(arr_full)
                elif method == "yeo":
                    pt = PowerTransformer(
                        method="yeo-johnson", standardize=True)
                    arr = pt.fit_transform(arr_full.reshape(-1, 1)).flatten()
                else:  # quantile
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=0)
                    arr = qt.fit_transform(arr_full.reshape(-1, 1)).flatten()

                sample = arr if arr.size <= 5000 else np.random.choice(
                    arr, 5000, replace=False)
                pval = 0.0
                if sample.size >= 3:
                    try:
                        pval = float(stats.shapiro(sample)[1])
                    except:
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
        self._report_lines.append("\nSTEP 6: Extra Transformation (Numeric)")
        for col in self.numeric_cols:
            orig = self._df[col]
            best, scores = self._evaluate_transform(orig)
            self.report["transform"][col] = {"chosen": best, "scores": scores}
            self._report_lines.append(
                f"  • {col:20s} → {best}, scores={scores}")
            if best == "log1p":
                self._df[col] = np.log1p(orig)
                self._report_lines.append(f"    • Applied log1p to '{col}'.")
            elif best == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                self._df[col] = pt.fit_transform(
                    orig.values.reshape(-1, 1)).flatten()
                self._report_lines.append(
                    f"    • Applied Yeo-Johnson to '{col}'.")
            elif best == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal", random_state=0)
                self._df[col] = qt.fit_transform(
                    orig.values.reshape(-1, 1)).flatten()
                self._report_lines.append(
                    f"    • Applied Quantile→Normal to '{col}'.")
            else:
                self._report_lines.append(
                    f"    • No transformation for '{col}'.")

    # ------------------------------------------------------------
    # 6) Categorical missing-value imputation
    # ------------------------------------------------------------
    def _random_sample_impute_cat(self, orig: pd.Series) -> pd.Series:
        arr = orig.copy().astype(object)
        nonnull = orig.dropna().astype(object).values
        if len(nonnull) == 0:
            return arr.fillna("__MISSING__")
        mask = arr.isna()
        arr.loc[mask] = np.random.choice(
            nonnull, size=mask.sum(), replace=True)
        return arr

    def _evaluate_impute_cat(self, orig: pd.Series) -> Tuple[str, Dict[str, float]]:
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
            score = 1.0 - tvd
            candidates["mode"] = arr
            scores["mode"] = score
        except:
            pass

        # constant_other
        try:
            arr = orig.fillna("__MISSING__").astype(object)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            score = 1.0 - tvd
            candidates["constant_other"] = arr
            scores["constant_other"] = score
        except:
            pass

        # random_sample
        try:
            arr = self._random_sample_impute_cat(orig)
            freq_after = arr.value_counts(normalize=True)
            common = set(freq_before.index).intersection(set(freq_after.index))
            tvd = float(
                np.sum(abs(freq_before.loc[list(common)] - freq_after.loc[list(common)])))
            score = 1.0 - tvd
            candidates["random_sample"] = arr
            scores["random_sample"] = score
        except:
            pass

        if not scores:
            return "fallback_constant", {"fallback_constant": 0.0}
        best = max(scores, key=scores.get)
        return best, scores

    def _impute_missing_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        self._report_lines.append(
            "\nSTEP 7: Categorical Missing-Value Imputation")
        for col in self.categorical_cols:
            orig = df[col].astype(object)
            n_missing = int(orig.isna().sum())
            if n_missing == 0:
                self.report["categorical_imputation"][col] = {
                    "chosen": "none", "note": "no missing"}
                self._report_lines.append(f"  • {col:20s} → no missing")
                continue

            best, scores = self._evaluate_impute_cat(orig)
            self.report["categorical_imputation"][col] = {
                "chosen": best, "scores": scores}
            self._report_lines.append(
                f"  • {col:20s} → {best}, scores={scores}")
            if best == "mode":
                imp = SimpleImputer(strategy="most_frequent")
                df[col] = pd.Series(imp.fit_transform(
                    orig.values.reshape(-1, 1)).flatten(), index=orig.index).astype(object)
                self._report_lines.append(
                    f"    • Applied mode imputation to '{col}'.")
            elif best == "constant_other":
                df[col] = orig.fillna("__MISSING__").astype(object)
                self._report_lines.append(
                    f"    • Filled nulls with '__MISSING__' in '{col}'.")
            elif best == "random_sample":
                df[col] = self._random_sample_impute_cat(orig)
                self._report_lines.append(
                    f"    • Random-sample imputed '{col}'.")
            else:
                df[col] = orig.fillna("__MISSING__").astype(object)
                self._report_lines.append(
                    f"    • Fallback: filled nulls with '__MISSING__' for '{col}'.")
        return df

    # ------------------------------------------------------------
    # 7) Rare-category grouping
    # ------------------------------------------------------------
    def _group_rare_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        self._report_lines.append("\nSTEP 8: Rare-Category Grouping")
        for col in self.categorical_cols:
            freq = df[col].value_counts(normalize=True)
            rare = set(freq[freq < self.RARE_FREQ_CUTOFF].index)
            if not rare:
                self.report["rare_categories"][col] = []
                self._report_lines.append(f"  • {col:20s} → no rare")
                continue
            self.report["rare_categories"][col] = list(rare)
            self._report_lines.append(
                f"  • {col:20s} → grouping {len(rare)} rare")
            df[col] = df[col].apply(lambda x: "__RARE__" if x in rare else x)
        return df

    # ------------------------------------------------------------
    # 8) Categorical encoding variants
    # ------------------------------------------------------------
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

    def _calculate_categorical_entropy(self, s: pd.Series) -> float:
        freqs = s.value_counts(normalize=True).values
        return float(scipy_entropy(freqs))

    def _encode_categorical_variants(self):
        self._report_lines.append("\nSTEP 9: Categorical Encoding Variants")
        base = self._df.copy()
        n_rows = len(base)

        # 9a) LINEAR
        linear_onehot, linear_freq, linear_sugg = [], [], []
        for col in self.categorical_cols:
            uniq = base[col].nunique()
            frac = uniq / n_rows if n_rows > 0 else 0.0
            ent = self._calculate_categorical_entropy(base[col].dropna())
            if frac <= self.ONEHOT_MAX_UNIQ:
                linear_onehot.append(col)
            else:
                if frac <= self.KNOWNALLY_MAX_UNIQ:
                    linear_freq.append(col)
                else:
                    linear_sugg.append(
                        f"  • [LINEAR] '{col}' frac={frac:.2f} > {self.ULTRAHIGH_UNIQ}: suggest target encoding."
                    )

        oh = self._onehot_encode(
            base, linear_onehot) if linear_onehot else pd.DataFrame(index=base.index)
        freq_df = self._frequency_encode(
            base, linear_freq) if linear_freq else pd.DataFrame(index=base.index)
        df_lin = pd.concat(
            [base.drop(columns=self.categorical_cols), oh, freq_df], axis=1)
        self.report["encoding"]["linear"] = {
            "onehot": linear_onehot,
            "frequency": linear_freq,
            "suggestions": linear_sugg,
        }
        self._report_lines.append(
            f"  • [LINEAR] onehot={linear_onehot}, freq={linear_freq}")
        for s in linear_sugg:
            self._report_lines.append(s)

        # 9b) TREE
        tree_onehot, tree_ordinal, tree_freq, tree_sugg = [], [], [], []
        for col in self.categorical_cols:
            uniq = base[col].nunique()
            frac = uniq / n_rows if n_rows > 0 else 0.0
            if frac <= self.ONEHOT_MAX_UNIQ:
                tree_onehot.append(col)
            elif frac <= self.ORDINAL_MAX_UNIQ:
                tree_ordinal.append(col)
            elif frac <= self.KNOWNALLY_MAX_UNIQ:
                tree_freq.append(col)
            else:
                tree_sugg.append(
                    f"  • [TREE] '{col}' frac={frac:.2f} > {self.ULTRAHIGH_UNIQ}: suggest target encoding."
                )

        df_tree = base.copy()
        if tree_onehot:
            oh2 = self._onehot_encode(base, tree_onehot)
            df_tree = df_tree.drop(columns=tree_onehot)
            df_tree = pd.concat([df_tree, oh2], axis=1)
        if tree_ordinal:
            ord_df = self._ordinal_encode(df_tree, tree_ordinal)
            df_tree = df_tree.drop(columns=tree_ordinal)
            df_tree = pd.concat([df_tree, ord_df], axis=1)
        if tree_freq:
            freq_df2 = self._frequency_encode(df_tree, tree_freq)
            df_tree = df_tree.drop(columns=tree_freq)
            df_tree = pd.concat([df_tree, freq_df2], axis=1)

        self.report["encoding"]["tree"] = {
            "onehot": tree_onehot,
            "ordinal": tree_ordinal,
            "frequency": tree_freq,
            "suggestions": tree_sugg,
        }
        self._report_lines.append(
            f"  • [TREE] onehot={tree_onehot}, ordinal={tree_ordinal}, freq={tree_freq}")
        for s in tree_sugg:
            self._report_lines.append(s)

        # 9c) KNN (frequency for all)
        freq_all = self._frequency_encode(
            base, self.categorical_cols) if self.categorical_cols else pd.DataFrame(index=base.index)
        df_knn = pd.concat(
            [base.drop(columns=self.categorical_cols), freq_all], axis=1)

        self.report["encoding"]["knn"] = {
            "frequency_all": self.categorical_cols}
        self._report_lines.append(
            f"  • [KNN] freq-encode all: {self.categorical_cols}")

        # Write CSVs
        Path("processed_linear.csv").write_text(
            "")  # placeholder ensure path exists
        Path("processed_tree.csv").write_text("")
        Path("processed_knn.csv").write_text("")
        df_lin.to_csv("processed_linear.csv", index=False)
        df_tree.to_csv("processed_tree.csv", index=False)
        df_knn.to_csv("processed_knn.csv", index=False)
        self._report_lines.append(
            "  • Wrote processed_linear.csv, processed_tree.csv, processed_knn.csv")

    # ------------------------------------------------------------
    # 9) PCA on numeric
    # ------------------------------------------------------------
    def _apply_pca(self):
        self._report_lines.append("\nSTEP 10: PCA on Numeric Columns")
        X = self._df[self.numeric_cols].copy()
        if X.isna().any(axis=None):
            self._report_lines.append("  • PCA aborted: NaNs present")
            self.report["pca"] = {"note": "skipped (NaNs)"}
            return

        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < 5 * n_feats:
            self._report_lines.append(
                f"  • PCA aborted: too few (n={n_samples}, p={n_feats})"
            )
            self.report["pca"] = {"note": "skipped (insufficient data)"}
            return

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.values)
        full_pca = PCA().fit(X_std)
        cumvar = np.cumsum(full_pca.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, self.pca_variance_threshold) + 1)

        self.report["pca"] = {"n_components": n_comp,
                              "cumvar": float(cumvar[n_comp - 1])}
        self._report_lines.append(
            f"  • PCA chooses {n_comp}/{n_feats} comps (cumvar={cumvar[n_comp-1]:.3f})")

        pca_model = PCA(n_components=n_comp)
        X_reduced = pca_model.fit_transform(X_std)
        cols = [f"PC{i+1}" for i in range(n_comp)]
        df_reduced = pd.DataFrame(
            X_reduced, columns=cols, index=self._df.index)

        self._df = pd.concat(
            [self._df.drop(columns=self.numeric_cols), df_reduced], axis=1)
        self.pca_model = pca_model
        self._report_lines.append(
            "  • Replaced numeric cols with principal components")

    # ------------------------------------------------------------
    # Public API: fit / transform
    # ------------------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        if isinstance(X, np.ndarray):
            raise ValueError("Input must be a pandas DataFrame")
        df = X.copy()
        if self.target and self.target in df.columns:
            df = df.drop(columns=[self.target])
        self._df = df.copy()
        self.numeric_cols = df.select_dtypes(
            include=np.number).columns.tolist()
        self.categorical_cols = [
            c for c in df.columns if c not in self.numeric_cols]
        self.non_numeric_cols = self.categorical_cols.copy()
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            raise ValueError("Input must be a pandas DataFrame")
        self._df = X.copy()
        if self.target and self.target in self._df.columns:
            self._df = self._df.drop(columns=[self.target])

        # 1) Drop duplicates
        self._df = self._drop_duplicates(self._df)

        # 2) Numeric missing imputation
        if self.numeric_cols:
            self._df = self._impute_missing_numeric(self._df)

        # 3) Outlier detection & removal
        if self.numeric_cols:
            self._detect_outliers()

        # 4) Scaling
        if self.numeric_cols:
            self._apply_scaling()

        # 5) Extra transformation
        if self.numeric_cols:
            self._apply_transform()

        # 6) Categorical missing imputation
        if self.categorical_cols:
            self._df = self._impute_missing_categorical(self._df)

        # 7) Rare grouping
        if self.categorical_cols:
            self._df = self._group_rare_categories(self._df)

        # 8) Categorical encoding variants (writes CSVs)
        if self.categorical_cols:
            self._encode_categorical_variants()

        # 9) PCA on numeric
        if self.numeric_cols:
            self._apply_pca()

        # Write analysis report
        with open("analysis_report.txt", "w") as f:
            f.write("\n".join(self._report_lines))

        return self._df.copy()

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def print_report(self):
        print("\n" + "=" * 80)
        print("✅ Data Quality Pipeline Report")
        print("=" * 80)
        for line in self._report_lines:
            print(line)
        print("\n" + "=" * 80 + "\n")


# =============================================================================
# 2) EDA (Basic + Advanced)
# =============================================================================

class EDA:
    """
    Phase 4: Exploratory Data Analysis

    - 4A: Univariate stats & plots
    - 4B: Bivariate tests & visuals
    - 4C: Multivariate diagnostics
    - Generates CSV summaries & PNGs under reports/eda/
    """

    def __init__(self, mode: str = "all", target: Optional[str] = None):
        """
        mode: "uva", "bva", "mva", or "all"
        target: column name for bivariate / leakage checks
        """
        self.mode = mode.lower()
        self.target = target

    def run_univariate(self, df: pd.DataFrame):
        out_dir = Path("reports/eda/uva")
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics = []
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            series = df[col].dropna()
            mean = series.mean()
            median = series.median()
            var = series.var()
            sd = series.std()
            skew = series.skew()
            kurt = series.kurtosis()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            # Normality tests
            try:
                shapiro_p = stats.shapiro(series)[1]
            except:
                shapiro_p = np.nan
            try:
                dag_p = stats.normaltest(series)[1]
            except:
                dag_p = np.nan
            try:
                jb_p = stats.jarque_bera(series)[1]
            except:
                jb_p = np.nan
            try:
                ad_p = stats.anderson(
                    series, dist="norm").significance_level[0]
            except:
                ad_p = np.nan

            metrics.append({
                "feature": col,
                "mean": mean, "median": median, "var": var, "sd": sd,
                "skew": skew, "kurt": kurt, "iqr": iqr,
                "shapiro_p": shapiro_p, "dagostino_p": dag_p,
                "jb_p": jb_p, "ad_p": ad_p
            })

            # Plot: histogram + KDE, boxplot, QQ-plot
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            sns.histplot(series, kde=True, ax=axs[0])
            axs[0].set_title(f"Histogram + KDE: {col}")
            sns.boxplot(x=series, ax=axs[1])
            axs[1].set_title(f"Boxplot: {col}")
            stats.probplot(series, dist="norm", plot=axs[2])
            axs[2].set_title(f"QQ-plot: {col}")
            plt.tight_layout()
            fig.savefig(out_dir / f"{col}_uva.png")
            plt.close(fig)

        pd.DataFrame(metrics).to_csv(
            "reports/eda/univariate_summary.csv", index=False)

    def run_bivariate(self, df: pd.DataFrame):
        out_dir = Path("reports/eda/bva")
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        results = []

        # num-num
        for i, x in enumerate(num_cols):
            for y in num_cols[i+1:]:
                xvals = df[x].dropna()
                yvals = df[y].dropna()
                common_idx = xvals.index.intersection(yvals.index)
                xvals = xvals.loc[common_idx]
                yvals = yvals.loc[common_idx]
                if len(xvals) < 2:
                    continue
                # Pearson
                try:
                    pearson_r, pearson_p = stats.pearsonr(xvals, yvals)
                except:
                    pearson_r, pearson_p = np.nan, np.nan
                # Spearman
                try:
                    spear_r, spear_p = stats.spearmanr(xvals, yvals)
                except:
                    spear_r, spear_p = np.nan, np.nan
                # Kendall
                try:
                    kend_r, kend_p = stats.kendalltau(xvals, yvals)
                except:
                    kend_r, kend_p = np.nan, np.nan
                r2 = pearson_r**2 if not np.isnan(pearson_r) else np.nan
                results.append({
                    "pair": f"{x}___{y}",
                    "pearson_r": pearson_r, "pearson_p": pearson_p,
                    "spearman_r": spear_r, "spearman_p": spear_p,
                    "kendall_r": kend_r, "kendall_p": kend_p,
                    "r2": r2,
                    "type": "num-num"
                })
                # Jointplot
                sns.jointplot(x=x, y=y, data=df, kind="reg").fig.savefig(
                    out_dir / "plots" / f"{x}_{y}_joint.png")
                plt.close()

        # num-cat (2 groups if binary, else ANOVA / Kruskal)
        if self.target and self.target in df.columns:
            tgt = df[self.target]
            for x in num_cols:
                yvals = tgt
                xvals = df[x]
                common = xvals.dropna().index.intersection(yvals.dropna().index)
                xvals = xvals.loc[common]
                yvals = yvals.loc[common]
                groups = yvals.unique()
                if len(groups) == 2:
                    # Welch’s t-test
                    g0 = xvals[yvals == groups[0]]
                    g1 = xvals[yvals == groups[1]]
                    try:
                        tstat, tp = stats.ttest_ind(g0, g1, equal_var=False)
                    except:
                        tstat, tp = np.nan, np.nan
                    # Mann-Whitney
                    try:
                        mw_u, mw_p = stats.mannwhitneyu(g0, g1)
                    except:
                        mw_u, mw_p = np.nan, np.nan
                    d = (g0.mean() - g1.mean()) / \
                        ((g0.std() + g1.std())/2 + 1e-9)
                    results.append({
                        "pair": f"{x}___{self.target}",
                        "t_stat": tstat, "t_p": tp,
                        "mw_u": mw_u, "mw_p": mw_p,
                        "cohens_d": d,
                        "type": "num-binary_cat"
                    })
                    sns.boxplot(x=yvals, y=xvals).figure.savefig(
                        out_dir / "plots" / f"{x}_{self.target}_box.png")
                    plt.close()
                elif len(groups) > 2:
                    # ANOVA
                    try:
                        arrays = [xvals[yvals == g] for g in groups]
                        fstat, fp = stats.f_oneway(*arrays)
                    except:
                        fstat, fp = np.nan, np.nan
                    # Kruskal-Wallis
                    try:
                        kw_h, kw_p = stats.kruskal(*arrays)
                    except:
                        kw_h, kw_p = np.nan, np.nan
                    results.append({
                        "pair": f"{x}___{self.target}",
                        "f_stat": fstat, "f_p": fp,
                        "kw_h": kw_h, "kw_p": kw_p,
                        "type": "num-multicat"
                    })
                    sns.violinplot(x=yvals, y=xvals).figure.savefig(
                        out_dir / "plots" / f"{x}_{self.target}_violin.png")
                    plt.close()

        # cat-cat
        for i, x in enumerate(cat_cols):
            for y in cat_cols[i+1:]:
                table = pd.crosstab(df[x], df[y])
                try:
                    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(table)
                except:
                    chi2_stat, chi2_p = np.nan, np.nan
                results.append({
                    "pair": f"{x}___{y}",
                    "chi2_stat": chi2_stat, "chi2_p": chi2_p,
                    "type": "cat-cat"
                })
                # Mosaic plot skipped (too verbose to code here)

        pd.DataFrame(results).to_csv(
            "reports/eda/bivariate_summary.csv", index=False)

        # Correlation heatmap
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="vlag")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("reports/eda/bva/plots/correlation_heatmap.png")
        plt.close()

    def run_multivariate(self, df: pd.DataFrame):
        out_dir = Path("reports/eda/mva")
        out_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # VIF
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        X = df[numeric_cols].dropna()
        vif_data = []
        for i, col in enumerate(numeric_cols):
            try:
                vif = variance_inflation_factor(X.values, i)
            except:
                vif = np.nan
            vif_data.append({"feature": col, "VIF": vif})
        pd.DataFrame(vif_data).to_csv("reports/eda/vif.csv", index=False)

        # Mardia’s multivariate normality (skewness & kurtosis)
        from pingouin import multivariate_normality
        try:
            mardia_skew, m_p = multivariate_normality(X, alpha=0.05)
            results["mardia_skew"] = m_p
        except:
            results["mardia_skew"] = np.nan

        with open("reports/eda/mva/mva_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        # PCA scree plot
        pca = PCA().fit(X)
        scree = pca.explained_variance_ratio_
        plt.figure()
        plt.plot(np.arange(1, len(scree) + 1), scree.cumsum(), marker="o")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Scree Plot")
        plt.grid()
        plt.savefig("reports/eda/mva/pca_scree.png")
        plt.close()

        # Correlation dendrogram (clustermap)
        sns.clustermap(X.corr(), cmap="vlag", figsize=(10, 10))
        plt.savefig("reports/eda/mva/corr_dendrogram.png")
        plt.close()

    def run(self, df: pd.DataFrame):
        Path("reports/eda/uva/plots").mkdir(parents=True, exist_ok=True)
        Path("reports/eda/bva/plots").mkdir(parents=True, exist_ok=True)
        Path("reports/eda/mva/plots").mkdir(parents=True, exist_ok=True)

        if self.mode in ("uva", "all"):
            self.run_univariate(df)
        if self.mode in ("bva", "all"):
            self.run_bivariate(df)
        if self.mode in ("mva", "all"):
            self.run_multivariate(df)


# =============================================================================
# 3) ProbabilisticAnalysis
# =============================================================================

class ProbabilisticAnalysis:
    """
    Phase 4½: Probabilistic Analysis

      1) Impute missing (mice, knn, simple)
      2) Fit best‐fit univariate distributions (AIC + KS)
      3) Shannon entropy (categorical & numeric)
      4) Mutual information scores (class/reg)
      5) Conditional probability tables (categorical vs. target)
      6) PIT transform (numeric)
      7) Quantile transform → normal (numeric)
      8) Bayesian group comparison (mean + 95% CI per group)
      9) Predictive intervals (quantile)
     10) Copula modeling (GaussianMultivariate)

    Outputs under `reports/prob/…`:
      - distributions.json
      - entropy.json
      - mi_scores.json
      - cond_prob_tables/*.csv
      - pit_transformed.csv
      - quantile_transformed.csv
      - bayesian_group.csv
      - copula_model.json
      - diagnostic plots…
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.imputed_data: Optional[pd.DataFrame] = None
        self.distributions: Dict[str, Dict] = {}
        Path("reports/prob/cond_prob_tables").mkdir(parents=True, exist_ok=True)

    def impute_missing(self, method="mice") -> pd.DataFrame:
        numeric_data = self.df.select_dtypes(include=np.number)
        if method == "mice":
            imputer = IterativeImputer(estimator=BayesianRidge(
            ), sample_posterior=True, random_state=0, max_iter=10)
        elif method == "knn":
            imputer = KNNImputer()
        else:
            imputer = SimpleImputer(strategy="mean")
        imputed = pd.DataFrame(imputer.fit_transform(
            numeric_data), columns=numeric_data.columns)
        self.imputed_data = imputed
        imputed.to_csv("reports/prob/imputed_numeric.csv", index=False)
        return imputed

    def fit_all_distributions(self) -> Dict[str, Tuple[str, float, float, float]]:
        results = {}
        for col in self.df.select_dtypes(include=np.number):
            values = self.df[col].dropna()
            dist_list = ["norm", "lognorm", "gamma", "beta", "weibull_min"]
            best = None
            best_aic = np.inf
            for dist_name in dist_list:
                dist = getattr(stats, dist_name)
                try:
                    params = dist.fit(values)
                    logpdf = dist.logpdf(values, *params)
                    aic = 2 * len(params) - 2 * float(np.sum(logpdf))
                    ks_stat, ks_p = stats.kstest(
                        values, dist_name, args=params)
                    if aic < best_aic:
                        best_aic = aic
                        best = (dist_name, aic, float(ks_stat), float(ks_p))
                except:
                    continue
            if best is not None:
                results[col] = {
                    "distribution": best[0],
                    "AIC": best[1],
                    "KS_stat": best[2],
                    "KS_p": best[3],
                }
        with open("reports/prob/distributions.json", "w") as f:
            json.dump(results, f, indent=2)
        self.distributions = results
        return results

    def shannon_entropy(self) -> Dict[str, float]:
        ent = {}
        for col in self.df.columns:
            if self.df[col].dtype == object or self.df[col].nunique() < 20:
                probs = self.df[col].value_counts(normalize=True)
                ent[col] = float(-np.sum(probs * np.log(probs + 1e-9)))
            else:
                hist, _ = np.histogram(
                    self.df[col].dropna(), bins=20, density=True)
                ent[col] = float(stats.entropy(hist + 1e-9))
        with open("reports/prob/entropy.json", "w") as f:
            json.dump(ent, f, indent=2)
        return ent

    def mutual_info_scores(self, target: str) -> Dict[str, float]:
        X = pd.get_dummies(self.df.drop(columns=[target]))
        y = self.df[target]
        if y.nunique() <= 10:
            mi = mutual_info_classif(X, y)
        else:
            mi = mutual_info_regression(X, y)
        mi_series = dict(zip(X.columns, mi.tolist()))
        with open("reports/prob/mi_scores.json", "w") as f:
            json.dump(mi_series, f, indent=2)
        return mi_series

    def conditional_probability_tables(self, target: str) -> Dict[str, pd.DataFrame]:
        tables = {}
        for col in self.df.select_dtypes(include=object):
            ct = pd.crosstab(self.df[col], self.df[target], normalize="index")
            ct.to_csv(f"reports/prob/cond_prob_tables/{col}_cond_prob.csv")
            tables[col] = ct
        return tables

    def pit_transform(self) -> pd.DataFrame:
        transformed = {}
        for col in self.df.select_dtypes(include=np.number):
            ecdf = ECDF(self.df[col].dropna())
            transformed[col] = self.df[col].map(lambda x: float(ecdf(x)))
        df_pit = pd.DataFrame(transformed)
        df_pit.to_csv("reports/prob/pit_transformed.csv", index=False)
        return df_pit

    def quantile_transform(self) -> pd.DataFrame:
        qt = QuantileTransformer(output_distribution="normal")
        arr = qt.fit_transform(self.df.select_dtypes(include=np.number))
        df_qt = pd.DataFrame(
            arr, columns=self.df.select_dtypes(include=np.number).columns)
        df_qt.to_csv("reports/prob/quantile_transformed.csv", index=False)
        return df_qt

    def bayesian_group_comparison(self, feature: str, target: str) -> pd.DataFrame:
        groups = self.df.groupby(feature)[target]
        stats_dict = {}
        for name, group in groups:
            mu = float(group.mean())
            ci = np.percentile(group, [2.5, 97.5]).tolist()
            stats_dict[name] = {"mean": mu, "CI_2.5": ci[0], "CI_97.5": ci[1]}
        df_bg = pd.DataFrame(stats_dict).T
        df_bg.to_csv("reports/prob/bayesian_group_comparison.csv")
        return df_bg

    def predictive_intervals(self, feature: str, alpha: float = 0.05) -> Tuple[float, float]:
        values = self.df[feature].dropna()
        lower = float(np.percentile(values, 100 * alpha / 2))
        upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
        with open("reports/prob/predictive_intervals.json", "w") as f:
            json.dump({feature: [lower, upper]}, f, indent=2)
        return lower, upper

    def copula_modeling(self) -> GaussianMultivariate:
        model = GaussianMultivariate()
        num_df = self.df.select_dtypes(include=np.number).dropna()
        model.fit(num_df)
        model.to_dict()  # get JSON-serializable
        with open("reports/prob/copula_model.json", "w") as f:
            json.dump(model.to_dict(), f, indent=2)
        return model

    def qq_pp_plots(self, feature: str):
        values = self.df[feature].dropna()
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        stats.probplot(values, dist="norm", plot=axs[0])
        axs[0].set_title("QQ Plot")
        sorted_data = np.sort(values)
        cdf = ECDF(sorted_data)
        axs[1].plot(sorted_data, cdf(sorted_data), label="Empirical")
        axs[1].plot(sorted_data, stats.norm.cdf(sorted_data), label="Normal")
        axs[1].legend()
        axs[1].set_title("PP Plot")
        plt.tight_layout()
        plt.savefig(f"reports/prob/{feature}_qqpp.png")
        plt.close()

    def detect_outliers(self) -> pd.Series:
        numeric = self.df.select_dtypes(include=np.number).dropna()
        cov = np.cov(numeric.T)
        inv_covmat = np.linalg.inv(cov)
        mean = numeric.mean().values
        diff = numeric - mean
        md = np.sqrt(np.diag(diff @ inv_covmat @ diff.T))
        md_series = pd.Series(md, index=numeric.index)
        md_series.to_csv("reports/prob/mahalanobis_distances.csv")
        return md_series

    def feature_importance(self, target: str) -> pd.Series:
        X = pd.get_dummies(self.df.drop(columns=[target]))
        y = self.df[target]
        if y.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=100)
        else:
            model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        perm = permutation_importance(model, X, y, n_repeats=10)
        imp_series = pd.Series(perm.importances_mean,
                               index=X.columns).sort_values(ascending=False)
        imp_series.to_csv("reports/prob/feature_importance.csv")
        return imp_series

    def diagnostic_plots(self, feature: str):
        values = self.df[feature].dropna()
        plt.figure(figsize=(10, 5))
        sns.histplot(values, kde=True, stat="density",
                     label="Empirical", color="blue")
        x = np.linspace(values.min(), values.max(), 100)
        if feature in self.distributions:
            dist_name = self.distributions[feature]["distribution"]
            dist = getattr(stats, dist_name)
            params = dist.fit(values)
            plt.plot(x, dist.pdf(x, *params),
                     label=f"Fitted {dist_name}", color="red")
        plt.title(f"Histogram + Fitted PDF for {feature}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"reports/prob/{feature}_diagnostic.png")
        plt.close()

    def run(self, df: pd.DataFrame, target: str):
        Path("reports/prob").mkdir(parents=True, exist_ok=True)

        self.df = df.copy()
        self.impute_missing("mice")
        self.fit_all_distributions()
        self.shannon_entropy()
        self.mutual_info_scores(target)
        self.conditional_probability_tables(target)
        self.pit_transform()
        self.quantile_transform()
        # For demonstration, run bayesian_group_comparison on first categorical vs target if exists
        cats = self.df.select_dtypes(include=object).columns.tolist()
        if cats:
            self.bayesian_group_comparison(cats[0], target)
        self.predictive_intervals(
            df.select_dtypes(include=np.number).columns[0])
        self.copula_modeling()
        # Produce a quick profile HTML
        try:
            profile = ProfileReport(
                self.df, title="Probabilistic Analysis Profile", minimal=True)
            profile.to_file("reports/prob/eda_profile.html")
        except:
            pass


# =============================================================================
# 4) Feature Selection
# =============================================================================

class FeatureSelect:
    """
    Phase 4½: Feature Selection

      1) Near‐zero‐variance filter (VarianceThreshold)
      2) Numeric correlation filter (|corr| ≥ threshold)
      3) Mutual information / F‐score filter (drop bottom quantile)

    Inputs:
       data: parquet path with cleaned/scaled data (numeric + categorical)
       target: name of target column
       nzv_threshold: float
       corr_threshold: float
       mi_quantile: float

    Outputs:
       data/processed/selected.parquet
       reports/feature/feature_audit.json
    """

    def __init__(
        self,
        nzv_threshold: float = 1e-5,
        corr_threshold: float = 0.95,
        mi_quantile: float = 0.10,
        verbose: bool = True,
    ):
        self.nzv_threshold = nzv_threshold
        self.corr_threshold = corr_threshold
        self.mi_quantile = mi_quantile
        self.verbose = verbose
        self.report: Dict[str, Union[int, List[str]]] = {}
        Path("reports/feature").mkdir(parents=True, exist_ok=True)

    def fit_transform(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        X = df.drop(columns=[target])
        y = df[target]

        # 1) NZV
        num = X.select_dtypes(include=np.number).columns.tolist()
        nzv = VarianceThreshold(threshold=self.nzv_threshold).fit(X[num])
        nzv_cols = [c for c, keep in zip(
            num, nzv.get_feature_names_out()) if keep]
        dropped_nzv = set(num) - set(nzv_cols)

        # 2) Corr filter
        corr = X[nzv_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        to_drop_corr = [col for col in upper.columns if any(
            upper[col] >= self.corr_threshold)]

        # 3) Mutual info / F-score filter
        num2 = [c for c in nzv_cols if c not in to_drop_corr]
        mi_cols = []
        if num2:
            if y.nunique() > 10:  # regression
                mi = mutual_info_regression(X[num2], y)
            else:
                mi = f_classif(X[num2], y)[0]
                mi[np.isnan(mi)] = 0
            mi_series = pd.Series(mi, index=num2)
            threshold_val = mi_series.quantile(self.mi_quantile)
            to_drop_mi = mi_series[mi_series < threshold_val].index.tolist()
        else:
            to_drop_mi = []

        drop_list = list(dropped_nzv | set(to_drop_corr) | set(to_drop_mi))
        X_sel = X.drop(columns=drop_list)

        self.report = {
            "n_initial_features": X.shape[1],
            "dropped_nzv": list(dropped_nzv),
            "dropped_corr": to_drop_corr,
            "dropped_mi": to_drop_mi,
            "n_remaining": X_sel.shape[1]
        }
        (Path("reports/feature") /
         "feature_audit.json").write_text(json.dumps(self.report, indent=2))
        X_sel.to_parquet("data/processed/selected.parquet", index=False)
        return pd.concat([X_sel, df[target]], axis=1)


# =============================================================================
# 5) FeatureEngineering
# =============================================================================

class FeatureEngineer:
    """
    Phase 5: Feature Engineering

      1) Filters (nzv, corr, mi) – already done in FeatureSelect
      2) Scaling/PowerTransform (numeric)
      3) Binning (quantile)
      4) Polynomial features & interactions
      5) Rare-category grouping
      6) Categorical encoding (one-hot, ordinal, target, woe, hash, freq)
      7) Text vectorization (tfidf, count, hashing)
      8) Datetime expansion (year, month, day, dow, hour)
      9) Cyclical encoding (sin/cos)
     10) Date deltas (days since reference)
     11) Aggregations (groupby roll-ups)
     12) SMOTE sampler (if needed)

    Outputs:
      - models/preprocessor.joblib
      - reports/feature/feature_shape.txt
      - reports/feature/feature_audit.json
    """

    def __init__(
        self,
        target: Optional[str] = None,
        numeric_scaler: str = "standard",
        numeric_power: Optional[str] = None,
        log_cols: Optional[List[str]] = None,
        quantile_bins: Optional[Dict[str, int]] = None,
        polynomial_degree: Optional[int] = None,
        interactions: bool = False,
        rare_threshold: Optional[float] = None,
        cat_encoding: str = "onehot",
        text_vectorizer: Optional[str] = None,
        text_cols: Optional[List[str]] = None,
        datetime_cols: Optional[List[str]] = None,
        cyclical_cols: Optional[Dict[str, int]] = None,
        date_delta_cols: Optional[Dict[str, str]] = None,
        aggregations: Optional[Dict[str, List[str]]] = None,
        sampler: Optional[str] = None,
        custom_steps: Optional[List[Callable[[
            pd.DataFrame], pd.DataFrame]]] = None,
        save_path: Union[str, Path] = "models/preprocessor.joblib",
        report_dir: Union[str, Path] = "reports/feature",
    ):
        self.target = target
        self.numeric_scaler = numeric_scaler
        self.numeric_power = numeric_power
        self.log_cols = log_cols or []
        self.quantile_bins = quantile_bins or {}
        self.polynomial_degree = polynomial_degree
        self.interactions = interactions
        self.rare_threshold = rare_threshold
        self.cat_encoding = cat_encoding
        self.text_vectorizer = text_vectorizer
        self.text_cols = text_cols or []
        self.datetime_cols = datetime_cols or []
        self.cyclical_cols = cyclical_cols or {}
        self.date_delta_cols = date_delta_cols or {}
        self.aggregations = aggregations or {}
        self.sampler = sampler
        self.custom_steps = custom_steps or []
        self.save_path = Path(save_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.pipe_ = None

    # --- Helper transformers ------------------------------------------------
    class FrequencyEncoder(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            X = pd.DataFrame(X)
            self.maps_ = {c: X[c].value_counts(normalize=True).to_dict()
                          for c in X.columns}
            return self

        def transform(self, X):
            X = pd.DataFrame(X).copy()
            for c in X:
                X[c] = X[c].map(self.maps_[c]).fillna(0.0)
            return X.values

    class Cyclical(BaseEstimator, TransformerMixin):
        def __init__(self, period: int):
            self.period = period

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            arr = np.array(X).astype(float)
            return np.c_[np.sin(2 * np.pi * arr / self.period),
                         np.cos(2 * np.pi * arr / self.period)]

    class TextLength(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            ser = X.iloc[:, 0].fillna("")
            return pd.DataFrame({
                f"{X.columns[0]}_n_chars": ser.str.len(),
                f"{X.columns[0]}_n_words": ser.str.split().str.len()
            }).values

    class DateDelta(BaseEstimator, TransformerMixin):
        def __init__(self, reference: str = "today"):
            self.ref = pd.Timestamp("now").normalize(
            ) if reference == "today" else pd.to_datetime(reference)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            ser = pd.to_datetime(X.iloc[:, 0])
            return (self.ref - ser).dt.days.to_frame().values

    class RareCategory(BaseEstimator, TransformerMixin):
        def __init__(self, th: float = 0.01):
            self.th = th
            self.map = {}

        def fit(self, X, y=None):
            X = pd.DataFrame(X)
            for c in X.columns:
                vc = X[c].value_counts(normalize=True)
                self.map[c] = set(vc[vc < self.th].index)
            return self

        def transform(self, X):
            X = pd.DataFrame(X).copy()
            for c in X.columns:
                X.loc[X[c].isin(self.map[c]), c] = "__RARE__"
            return X.values

    # --- Build ColumnTransformer --------------------------------------------
    def _build_pretransform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply near-zero variance, correlation, MI filters BEFORE heavy transforms."""
        keep = X.copy()
        # 1) NZV on numeric
        num = keep.select_dtypes(include=np.number).columns.tolist()
        if num:
            nzv = VarianceThreshold(threshold=1e-5).fit(keep[num])
            nzv_keep = [c for c, flag in zip(num, nzv.get_support()) if flag]
            drop_nzv = list(set(num) - set(nzv_keep))
            keep.drop(columns=drop_nzv, inplace=True)
        # 2) Correlation
        num2 = keep.select_dtypes(include=np.number).columns.tolist()
        if len(num2) > 1:
            corr = keep[num2].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
            drop_corr = [c for c in upper.columns if any(upper[c] >= 0.95)]
            keep.drop(columns=drop_corr, inplace=True)
        # 3) MI / F-score
        num3 = keep.select_dtypes(include=np.number).columns.tolist()
        if num3 and self.target:
            y = X[self.target]
            if y.nunique() > 10:
                mi = mutual_info_regression(X[num3], y, random_state=0)
            else:
                mi_arr = f_classif(X[num3], y)[0]
                mi_arr[np.isnan(mi_arr)] = 0
                mi = mi_arr
            mi_series = pd.Series(mi, index=num3)
            low = mi_series.quantile(0.10)
            drop_mi = mi_series[mi_series < low].index.tolist()
            keep.drop(columns=drop_mi, inplace=True)
        return keep

    def _build_column_transformer(self, X: pd.DataFrame) -> TransformerMixin:
        num = X.select_dtypes(include=np.number).columns.tolist()
        cat = X.select_dtypes(include=object).columns.tolist()
        txt = self.text_cols or []
        cat = [c for c in cat if c not in txt]

        # Numeric pipeline
        nsteps = []
        if self.log_cols:
            lt = [
                (f"log_{c}", TransformerMixin(), [c])
                for c in self.log_cols if c in num
            ]
            if lt:
                nsteps.append(("log",
                               ColumnTransformer([(name, PowerTransformer(), cols) for name, _, cols in lt], remainder="passthrough")))

        if self.numeric_power:
            nsteps.append(
                ("power", PowerTransformer(method=self.numeric_power)))

        scale_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "none": TransformerMixin()
        }
        if self.numeric_scaler != "none":
            nsteps.append(("scale", scale_map[self.numeric_scaler]))

        if self.quantile_bins:
            qsteps = [
                (f"bin_{c}", TransformerMixin(), [c])
                for c, b in self.quantile_bins.items() if c in num
            ]
            if qsteps:
                nsteps.append(("qbin", ColumnTransformer(
                    [(name, KBinsDiscretizer(n_bins=b, encode="ordinal", strategy="quantile"), [c])
                     for c, b in self.quantile_bins.items() if c in num],
                    remainder="passthrough"
                )))

        if self.polynomial_degree:
            nsteps.append(("poly",
                           TransformerMixin() if self.interactions else PolynomialFeatures(
                               self.polynomial_degree, include_bias=False)
                           ))

        num_pipe = Pipeline(nsteps) if nsteps else TransformerMixin()

        # Categorical pipeline
        csteps = []
        if self.rare_threshold:
            csteps.append(("rare", self.RareCategory(self.rare_threshold)))
        enc_map = {
            "onehot": TransformerMixin(),  # will handle manually
            "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            "target": TargetEncoder(),
            "woe": WOEEncoder(),
            "hash": HashingEncoder(),
            "freq": self.FrequencyEncoder(),
            "none": TransformerMixin()
        }
        csteps.append(("enc", enc_map[self.cat_encoding]))
        cat_pipe = Pipeline(csteps)

        # Assemble ColumnTransformer
        transformers = [("num", num_pipe, num), ("cat", cat_pipe, cat)]
        if self.text_vectorizer and txt:
            tv_map = {
                "tfidf": TfidfVectorizer,
                "count": CountVectorizer,
                "hashing": HashingVectorizer
            }
            for col in txt:
                transformers.append(
                    (f"text_{col}", tv_map[self.text_vectorizer](
                        max_features=100, ngram_range=(1, 2)), col)
                )
                transformers.append(
                    (f"textlen_{col}", self.TextLength(), [col]))

        # Cyclical & date-delta & datetime handled via custom transformers if provided
        return ColumnTransformer(transformers, remainder="drop")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.target and self.target in X.columns:
            y = y or X[self.target]
            X = X.drop(columns=[self.target])

        X_clean = self._build_pretransform(X)
        pre = self._build_column_transformer(X_clean)
        steps = [("pre", pre)]

        if self.interactions and self.polynomial_degree:
            steps.append(("poly_inter", TransformerMixin()))

        for fn in self.custom_steps or []:
            steps.append((fn.__name__, TransformerMixin()))

        if self.sampler == "smote" and y is not None and y.dtype.kind not in "iu":
            steps.append(("smote", SMOTE(random_state=0)))

        self.pipe_ = Pipeline(steps).fit(X_clean, y)

        audit = {
            "n_features_in": X.shape[1],
            "n_features_after_clean": X_clean.shape[1],
            "pipeline_steps": [name for name, _ in self.pipe_.steps],
        }
        (self.report_dir / "feature_audit.json").write_text(json.dumps(audit, indent=2))
        shape = self.pipe_.transform(X_clean).shape
        with open(self.report_dir / "feature_shape.txt", "w") as f:
            f.write(f"{shape}\n")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.target and self.target in X.columns:
            X = X.drop(columns=[self.target])
        return self.pipe_.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self.pipe_, self.save_path)
        sha = hashlib.sha256(self.save_path.read_bytes()).hexdigest()[:12]
        (self.save_path.with_suffix(".json")).write_text(
            json.dumps({"sha256": sha}, indent=2))


# =============================================================================
# 6) SplitAndBaseline
# =============================================================================

class SplitAndBaseline:
    """
    Phase 5½: Split & Baseline Benchmarking

      1) Train / Val / Test split (stratified optional)
      2) Baseline models (majority class or mean regressor)
      3) Sanity checks (duplicate index, leakage)
      4) Freeze preprocessor (StandardScaler on numeric only)

    Outputs:
      - data/splits/{train,val,test}.parquet
      - split_manifest.json
      - reports/baseline/baseline_metrics.json
      - models/preprocessor.joblib
      - models/preprocessor_manifest.json
    """

    def __init__(
        self,
        target: str,
        seed: int = 42,
        stratify: bool = False,
        oversample: bool = False,
    ):
        self.target = target
        self.seed = seed
        self.stratify = stratify
        self.oversample = oversample

        self.PROC = Path("data/processed/selected.parquet")
        self.SPLIT = Path("data/splits")
        self.SPLIT.mkdir(parents=True, exist_ok=True)
        self.REPORT = Path("reports/baseline")
        self.REPORT.mkdir(parents=True, exist_ok=True)
        self.MODEL = Path("models")
        self.MODEL.mkdir(exist_ok=True)

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(self.PROC)
        y = df[self.target]
        strat = y if self.stratify else None

        train, temp = train_test_split(
            df, test_size=0.3, random_state=self.seed, stratify=strat)
        val, test = train_test_split(
            temp, test_size=0.5, random_state=self.seed,
            stratify=strat.loc[temp.index] if self.stratify else None
        )

        if self.oversample and y.dtype.kind not in "fu":
            X_tr, y_tr = train.drop(columns=[self.target]), train[self.target]
            X_tr, y_tr = SMOTE(random_state=self.seed).fit_resample(X_tr, y_tr)
            train = pd.concat([X_tr, y_tr], axis=1)

        train.to_parquet(self.SPLIT / "train.parquet", index=False)
        val.to_parquet(self.SPLIT / "val.parquet", index=False)
        test.to_parquet(self.SPLIT / "test.parquet", index=False)

        manifest = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "seed": self.seed,
            "stratify": self.stratify,
            "oversample": self.oversample,
            "target": self.target,
            "rows": {"train": len(train), "val": len(val), "test": len(test)}
        }
        (self.SPLIT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
        return train, val, test

    def build_baseline(self, train: pd.DataFrame, test: pd.DataFrame):
        y_test = test[self.target]
        if y_test.dtype.kind in "iuf":
            pred = np.full_like(y_test, train[self.target].mean(), dtype=float)
            metrics = {
                "type": "mean_regressor",
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred))
            }
        else:
            majority = int(train[self.target].mode()[0])
            pred = np.full_like(y_test, majority)
            metrics = {
                "type": "majority_class",
                "majority_class": majority,
                "accuracy": float(accuracy_score(y_test, pred)),
                "f1": float(f1_score(y_test, pred, zero_division=0))
            }
        (self.REPORT / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))

    def sanity_checks(self):
        tr = pd.read_parquet(self.SPLIT / "train.parquet")
        te = pd.read_parquet(self.SPLIT / "test.parquet")
        dup = set(tr.index).intersection(te.index)
        assert not dup, f"Duplicate rows across splits: {len(dup)}"
        leaks = [
            c for c in tr.columns if c != self.target and tr[c].equals(tr[self.target])
        ]
        assert not leaks, f"Potential leakage columns: {leaks}"

    def freeze_preprocessor(self):
        df = pd.read_parquet(self.PROC)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        scaler = Pipeline([("scale", StandardScaler())])
        scaler.fit(df[num_cols])
        import joblib
        joblib.dump(scaler, self.MODEL / "preprocessor.joblib")
        sha = hashlib.sha256(
            (self.MODEL / "preprocessor.joblib").read_bytes()).hexdigest()[:12]
        (self.MODEL / "preprocessor_manifest.json").write_text(
            json.dumps({"sha256": sha}, indent=2))

    def run(self):
        train, val, test = self.split_data()
        self.build_baseline(train, test)
        self.sanity_checks()
        self.freeze_preprocessor()
        print("🟢 Split & baseline phase complete.")


# =============================================================================
# 7) MLProjectPipeline (Master Orchestrator)
# =============================================================================

class MLProjectPipeline:
    """
    Master orchestrator for Phases 2–5 of MLDLC (up through Split & Baseline).
    """

    def __init__(
        self,
        input_csv: str,
        target: str,
        seed: int = 42,
        stratify: bool = True,
        oversample: bool = False,
        verbose: bool = True,
    ):
        self.input_csv = Path(input_csv)
        self.target = target
        self.seed = seed
        self.stratify = stratify
        self.oversample = oversample
        self.verbose = verbose

    def run(self):
        # Create output directories
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/interim").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/processed/scaled_parquet_dummy").mkdir(parents=True, exist_ok=True)
        Path("data/processed/selected_parquet_dummy").mkdir(parents=True, exist_ok=True)
        Path("data/splits").mkdir(parents=True, exist_ok=True)
        Path("reports/eda").mkdir(parents=True, exist_ok=True)
        Path("reports/prob").mkdir(parents=True, exist_ok=True)
        Path("reports/feature").mkdir(parents=True, exist_ok=True)
        Path("reports/baseline").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)

        # Phase 2: Data Collection (assume input CSV already in data/raw)
        print("▶ Phase 2: Data Collection (input assumed at data/raw)")
        raw_df = pd.read_csv(self.input_csv)
        raw_df.to_parquet("data/raw/combined.parquet", index=False)

        # Phase 3: Data Quality & Preprocessing
        print("▶ Phase 3: Data Quality & Preprocessing")
        dq = DataQualityPipeline(
            target_column=self.target, verbose=self.verbose)
        processed_df = dq.fit_transform(raw_df)
        dq.print_report()
        processed_df.to_parquet("data/interim/clean.parquet", index=False)

        # Also save scaled numeric + categorical (before feature selection)
        processed_df.to_parquet("data/processed/scaled.parquet", index=False)

        # Phase 4: EDA
        print("▶ Phase 4: Exploratory Data Analysis")
        df_for_eda = pd.read_parquet("data/interim/clean.parquet")
        eda = EDA(mode="all", target=self.target)
        eda.run(df_for_eda)

        # Phase 4½: Probabilistic Analysis
        print("▶ Phase 4½: Probabilistic Analysis")
        prob = ProbabilisticAnalysis(raw_df)
        prob.run(raw_df, self.target)

        # Phase 4½: Feature Selection
        print("▶ Phase 4½: Feature Selection")
        feat_sel = FeatureSelect(
            nzv_threshold=1e-5, corr_threshold=0.95, mi_quantile=0.10, verbose=self.verbose)
        df_selected = feat_sel.fit_transform(processed_df, self.target)

        # Phase 5: Feature Engineering
        print("▶ Phase 5: Feature Engineering")
        fe = FeatureEngineer(
            target=self.target,
            numeric_scaler="robust",
            numeric_power="yeo",
            log_cols=["revenue"] if "revenue" in df_selected.columns else [],
            quantile_bins={"age": 4} if "age" in df_selected.columns else {},
            polynomial_degree=2,
            interactions=True,
            rare_threshold=0.01,
            cat_encoding="target",
            text_vectorizer="tfidf",
            text_cols=["review"] if "review" in df_selected.columns else [],
            datetime_cols=[
                "last_login"] if "last_login" in df_selected.columns else [],
            cyclical_cols={
                "hour": 24} if "hour" in df_selected.columns else {},
            date_delta_cols={
                "signup_date": "today"} if "signup_date" in df_selected.columns else {},
            aggregations={"customer_id": [
                "amount_mean", "amount_sum"]} if "customer_id" in df_selected.columns else {},
            sampler="smote",
            custom_steps=[],
            save_path="models/preprocessor.joblib",
            report_dir="reports/feature"
        )
        fe.fit(df_selected, df_selected[self.target])
        fe.save()

        # Phase 5½: Split & Baseline
        print("▶ Phase 5½: Split & Baseline Benchmark")
        sab = SplitAndBaseline(
            target=self.target,
            seed=self.seed,
            stratify=self.stratify,
            oversample=self.oversample
        )
        sab.run()

        print("\n✅ All preprocessing phases completed.")


# =============================================================================
# CLI entrypoint
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full MLDLC pipeline up through Split & Baseline.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input CSV")
    parser.add_argument("--target", "-t", required=True,
                        help="Name of target column")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-stratify", action="store_false",
                        dest="stratify", help="Disable stratification")
    parser.add_argument("--oversample", action="store_true",
                        help="Apply SMOTE on training split")
    args = parser.parse_args()

    pipeline = MLProjectPipeline(
        input_csv=args.input,
        target=args.target,
        seed=args.seed,
        stratify=args.stratify,
        oversample=args.oversample,
        verbose=True
    )
    pipeline.run()
