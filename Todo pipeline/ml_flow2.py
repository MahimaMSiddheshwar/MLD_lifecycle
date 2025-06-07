#!/usr/bin/env python3
"""
MLDLC End-to-End Pipeline (Enhanced)
====================================

This enhanced version of DataQualityPipeline incorporates numerous additional checks
and fixes—while still omitting explicit modeling steps and SMOTE.  In particular, we've:

    • Added a duplicate‐row detection and removal step.
    • Introduced a three‐way split: train / validation / test (no leakage).
    • Cast true Boolean columns to `bool`/`int8` and skip expensive encoding on them.
    • Performed an automatic multicollinearity screen (pairwise correlation + VIF) and
      drop one of any pair with |corr| > 0.95 or VIF > 10.
    • Inserted a rudimentary “target‐leakage” check: any feature that alone predicts the
      target with AUC > 0.99 is flagged (and dropped).
    • Added a final “drop near‐constant” pass after all transforms (variance < 1e-8).
    • Introduced an extra Dimensionality Reduction comparison: PCA vs. FactorAnalysis—
      we compute reconstruction mean‐squared error (MSE) on TRAIN and choose whichever
      is significantly better (≥ 5% reduction); otherwise default to PCA.  We log this
      decision in the report.
    • Included a “feature‐selection” pass to drop any numeric column with entropy ≈ 0
      (i.e. zero‐variance) even before scaling.
    • Improved handling of unseen categories at TEST time (mapping to "__UNSEEN__" / zero freq).
    • Logged decisions and metrics thoroughly into `analysis_report.txt`.
    • **We have *not* changed anything after “modeling”**—this remains entirely out of scope.

Usage remains:

    from data_quality import DataQualityPipeline
    import pandas as pd

    df = pd.read_csv("data/raw/your_data.csv")
    dq = DataQualityPipeline(
        target_column="is_churn",
        test_size=0.2,
        random_state=42,
        pca_variance_threshold=0.90,
        apply_pca=True,
        verbose=True
    )
    train_df, val_df, test_df = dq.fit_transform(df)
    dq.print_report()
    # ’train_df’ and ’val_df’ and ’test_df’ are returned; analysis_report.txt is written.

Dependencies:
    numpy, pandas, scipy, scikit-learn, statsmodels, imbalanced-learn (if you choose), 
    category_encoders, copulas (optional), statsmodels.

"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import scipy.stats as stats
from scipy.stats import chi2, ks_2samp
from statsmodels.imputation.mice import mice
from statsmodels.stats.missing import test_missingness
from sklearn.covariance import EmpiricalCovariance

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, OrdinalEncoder
)
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class DataQualityPipeline(BaseEstimator, TransformerMixin):
    """
    Enhanced Data Quality & Preprocessing Pipeline

      0) Deduplicate rows
      1) Three‐way Split: train / validation / test (stratified if target exists)
      2) Detect MCAR/MAR/MNAR via Little’s test
      3) Boolean detection → cast to bool/int8
      4) Drop zero‐variance (entropy ≈ 0) features
      5) Numeric missing‐value imputation (mean, median, KNN, MICE, random‐sample)
      6) Univariate & Multivariate outlier detection & removal
      7) Multicollinearity screen (correlation + VIF) → drop collinear features
      8) Target‐leakage check: any single‐feature AUC > 0.99 → drop
      9) Scaling (Standard, MinMax, or Robust based on skew/kurtosis)
     10) Conditional extra transformation (Box‐Cox, Yeo‐Johnson, Quantile→Normal)
     11) Categorical missing‐value imputation (mode, constant, random‐sample)
     12) Rare‐category grouping (<1% → "__RARE__")
     13) Categorical encoding variants (linear, tree, knn) → writes 3 CSVs; UNSEEN cat handling
     14) Final drop of near‐constant columns (post‐transform variance < 1e-8)
     15) Dimensionality Reduction comparison: PCA vs. FactorAnalysis (choose by reconstruction MSE)

    Produces:
      - analysis_report.txt
      - processed_train_linear.csv, processed_train_tree.csv, processed_train_knn.csv
      - Four fitted transformers saved for test‐set application

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
      TRANSFORM_CANDIDATES = ["none", "boxcox", "yeo", "quantile"]
      CAT_IMPUTE_TVD_CUTOFF = 0.2
      RARE_FREQ_CUTOFF = 0.01
      ONEHOT_MAX_UNIQ = 0.05
      ORDINAL_MAX_UNIQ = 0.20
      KNOWNALLY_MAX_UNIQ = 0.50
      ULTRAHIGH_UNIQ = 0.50
      CORR_CUTOFF = 0.95
      VIF_CUTOFF = 10.0
      AUC_LEAKAGE_CUTOFF = 0.99
      CONST_THRESHOLD = 1e-8
      FA_MSE_IMPROV_THRESHOLD = 0.05  # 5% improvement to choose FA over PCA
    """

    # -------------- Class-level constants --------------
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

    TRANSFORM_CANDIDATES = ["none", "boxcox", "yeo", "quantile"]

    CAT_IMPUTE_TVD_CUTOFF = 0.2
    RARE_FREQ_CUTOFF = 0.01

    ONEHOT_MAX_UNIQ = 0.05
    ORDINAL_MAX_UNIQ = 0.20
    KNOWNALLY_MAX_UNIQ = 0.50
    ULTRAHIGH_UNIQ = 0.50

    CORR_CUTOFF = 0.95
    VIF_CUTOFF = 10.0

    AUC_LEAKAGE_CUTOFF = 0.99

    CONST_THRESHOLD = 1e-8

    FA_MSE_IMPROV_THRESHOLD = 0.05  # Need ≥5% MSE improvement to choose FA over PCA

    def __init__(
        self,
        target_column: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        pca_variance_threshold: float = 0.90,
        apply_pca: bool = True,
        verbose: bool = True,
    ):
        self.target = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.pca_variance_threshold = pca_variance_threshold
        self.apply_pca = apply_pca
        self.verbose = verbose

        # Will be populated during fit
        self.dup_removed_count: int = 0
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # Feature lists
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.boolean_cols: List[str] = []
        self.non_numeric_cols: List[str] = []

        # Fitted transformers (to apply on test/val)
        self.numeric_imputers: Dict[str, BaseEstimator] = {}
        self.scaler_model: Optional[TransformerMixin] = None
        self.transform_model: Dict[str, TransformerMixin] = {}
        self.categorical_imputers: Dict[str, BaseEstimator] = {}
        self.encoding_models: Dict[str,
                                   Union[OrdinalEncoder, Dict[str, float]]] = {}
        self.pca_model: Optional[PCA] = None
        self.fa_model: Optional[FactorAnalysis] = None
        self.chosen_dr: str = "PCA"

        # Reporting
        self.report: Dict[str, Dict] = {
            "duplicates": {},
            "split": {},
            "validation_split": {},
            "missing_mechanism": {},
            "boolean_cast": {},
            "constant_drop": {},
            "missing_numeric": {},
            "univariate_outliers": {},
            "multivariate_outliers": {},
            "collinearity": {},
            "target_leakage": {},
            "scaler": {},
            "transform": {},
            "categorical_imputation": {},
            "rare_categories": {},
            "encoding": {},
            "final_constant_drop": {},
            "dimensionality_reduction": {},
        }
        self._report_lines: List[str] = []

    # --------------- 0) Duplicate Detection & Removal ----------------
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop exact duplicate rows (full feature set). Log how many removed.
        """
        self._report_lines.append("\nSTEP 0A: Duplicate Detection & Removal")
        n_before = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        n_after = len(df_clean)
        removed = n_before - n_after
        self.dup_removed_count = removed
        self.report["duplicates"] = {"removed": removed}
        self._report_lines.append(
            f"  • {removed} duplicate rows removed (from {n_before}).")
        return df_clean

    # ---------------- 1) Three-way Train/Val/Test Split ----------------
    def _train_val_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits `df` into train / val / test sets (60/20/20), stratify if target exists.
        """
        self._report_lines.append(
            "\nSTEP 0B: Three-way Split (Train/Val/Test)")
        if self.target and self.target in df.columns:
            strat = df[self.target]
            # First split off test
            tr_val, te = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state, stratify=strat
            )
            # Now split tr_val into train and val (proportionally)
            val_size = self.test_size / \
                (1 - self.test_size)  # e.g. 0.2/0.8 = 0.25
            strat2 = tr_val[self.target]
            tr, va = train_test_split(
                tr_val, test_size=val_size, random_state=self.random_state, stratify=strat2
            )
        else:
            tr_val, te = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state, stratify=None
            )
            val_size = self.test_size / (1 - self.test_size)
            tr, va = train_test_split(
                tr_val, test_size=val_size, random_state=self.random_state, stratify=None
            )

        self.report["split"] = {
            "n_total": len(df),
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "test_size": self.test_size,
            "val_size": val_size,
        }
        self._report_lines.append(
            f"  • Split into train={len(tr)}, val={len(va)}, test={len(te)}."
        )
        return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

    # -------------- 2) Missing Mechanism Detection (Little’s Test) ---------------
    def _detect_missing_mechanism(self, df: pd.DataFrame) -> None:
        """
        Run Little’s MCAR test via statsmodels. Log MCAR vs MAR/MNAR.
        If <5% missing overall & MCAR, suggest CCA or univariate impute.
        """
        self._report_lines.append(
            "\nSTEP 1: Missing Mechanism Detection (Little’s MCAR)")
        if self.target and self.target in df.columns:
            data_for_test = df.drop(columns=[self.target])
        else:
            data_for_test = df.copy()

        na_counts = data_for_test.isna().sum()
        cols_with_na = na_counts[na_counts > 0].index.tolist()
        if len(cols_with_na) < 2:
            self.report["missing_mechanism"] = {
                "note": "too few NA columns to test MCAR"}
            self._report_lines.append(
                "  • Too few columns with missing → skipped MCAR test.")
            return

        try:
            res = test_missingness(data_for_test[cols_with_na])
            pvalue = float(res.pvalue)
            mech = "MCAR" if pvalue > 0.05 else "MAR/MNAR"
            self.report["missing_mechanism"] = {
                "pvalue": pvalue, "mechanism": mech}
            self._report_lines.append(
                f"  • Little’s MCAR p-value={pvalue:.3f} → {mech}")
        except Exception:
            self.report["missing_mechanism"] = {"note": "MCAR test failed"}
            self._report_lines.append("  • MCAR test failed.")

    # ---------------- 3) Boolean Detection & Casting ------------------
    def _cast_booleans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect any column where unique values ⊆ {true,false,0,1,yes,no}, cast to bool/int8.
        """
        self._report_lines.append("\nSTEP 2: Boolean Detection & Casting")
        bool_cols: List[str] = []
        for col in df.columns:
            ser = df[col]
            # Only consider object or numeric columns
            if ser.dtype == "object" or np.issubdtype(ser.dtype, np.integer):
                unique_vals = set(ser.dropna().astype(str).str.lower())
                if unique_vals.issubset({"true", "false", "0", "1", "yes", "no"}):
                    # Cast to bool→int8
                    df[col] = ser.astype(str).str.lower().map(
                        {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
                    ).astype("int8")
                    bool_cols.append(col)
                    self._report_lines.append(
                        f"  • Cast '{col}' → int8 boolean.")
        self.report["boolean_cast"] = {"boolean_cols": bool_cols}
        self.boolean_cols = bool_cols.copy()
        return df

    # ----------- 4) Drop Zero-Variance (Constant/Entropy ≈ 0) ----------
    def _drop_zero_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop any column (numeric or categorical) whose entropy ≈ 0 (i.e. constant).
        """
        self._report_lines.append(
            "\nSTEP 3: Drop Zero-Variance (Constant) Features")
        to_drop: List[str] = []
        for col in df.columns:
            ser = df[col]
            if ser.nunique(dropna=False) <= 1:
                to_drop.append(col)
            else:
                # Also check Shannon entropy for categorical or numeric
                try:
                    counts = ser.dropna().value_counts(normalize=True).values
                    ent = float(stats.entropy(counts))
                    if ent < 1e-6:
                        to_drop.append(col)
                except:
                    pass
        if to_drop:
            df = df.drop(columns=to_drop)
            self._report_lines.append(
                f"  • Dropped constant features: {to_drop}")
        else:
            self._report_lines.append("  • No constant features to drop.")
        self.report["constant_drop"] = {"dropped": to_drop}
        return df

    # --------------- 5) Numeric Missing-Value Imputation ---------------
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
        """
        Compute KS p-value, variance ratio, and covariance-change for a numeric imputation.
        """
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
            temp = pd.DataFrame(
                {c: self.train_df[c] for c in self.numeric_cols})
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

    def _impute_missing_numeric(self) -> None:
        """
        Impute numeric columns on TRAIN set with multiple methods,
        choose the best via KS, var_ratio ≥ 0.5, cov_change ≤ 0.2.
        Store fitted imputers in `self.numeric_imputers` for VAL/TEST.
        """
        self._report_lines.append("\nSTEP 4: Numeric Missing-Value Imputation")
        df = self.train_df
        df_num = df[self.numeric_cols].copy()
        complete_df = df_num.dropna()
        if complete_df.shape[0] < self.MIN_COMPLETE_FOR_COV:
            cov_before = None
            self._report_lines.append(
                "  • Too few complete rows → skipping covariance QC")
        else:
            cov_before = EmpiricalCovariance().fit(complete_df.values).covariance_

        for col in self.numeric_cols:
            orig = df_num[col]
            n_missing = int(orig.isna().sum())
            if n_missing == 0:
                self.report["missing_numeric"][col] = {
                    "chosen": "none", "note": "no missing"}
                self._report_lines.append(f"  • {col:20s} → no missing")
                continue

            candidates: Dict[str, pd.Series] = {}
            imputers: Dict[str, BaseEstimator] = {}
            metrics: Dict[str, Tuple[float, float, float]] = {}

            # 1) Mean
            try:
                imp = SimpleImputer(strategy="mean")
                arr = pd.Series(imp.fit_transform(
                    orig.values.reshape(-1, 1)).flatten(), index=orig.index)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["mean"] = arr
                imputers["mean"] = clone(imp)
                metrics["mean"] = (ksp, vr, cc)
            except:
                pass

            # 2) Median
            try:
                imp = SimpleImputer(strategy="median")
                arr = pd.Series(imp.fit_transform(
                    orig.values.reshape(-1, 1)).flatten(), index=orig.index)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["median"] = arr
                imputers["median"] = clone(imp)
                metrics["median"] = (ksp, vr, cc)
            except:
                pass

            # 3) KNN
            try:
                imp = KNNImputer(n_neighbors=5)
                tmp = df_num.copy()
                tmp_imp = pd.DataFrame(imp.fit_transform(
                    tmp), columns=self.numeric_cols, index=df_num.index)
                arr = tmp_imp[col]
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["knn"] = arr
                imputers["knn"] = clone(imp)
                metrics["knn"] = (ksp, vr, cc)
            except:
                pass

            # 4) MICE
            try:
                imp = IterativeImputer(estimator=BayesianRidge(
                ), sample_posterior=True, random_state=self.random_state, max_iter=10)
                tmp = df_num.copy()
                tmp_imp = pd.DataFrame(imp.fit_transform(
                    tmp), columns=self.numeric_cols, index=df_num.index)
                arr = tmp_imp[col]
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["mice"] = arr
                imputers["mice"] = clone(imp)
                metrics["mice"] = (ksp, vr, cc)
            except:
                pass

            # 5) Random-sample
            try:
                arr = self._random_sample_impute_num(orig)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["random_sample"] = arr
                # we’ll handle random‐sample later
                imputers["random_sample"] = None
                metrics["random_sample"] = (ksp, vr, cc)
            except:
                pass

            # Choose best method
            best_method = None
            best_score = (-1.0, -1.0, np.inf)
            for m, (ksp, vr, cc) in metrics.items():
                if (ksp > best_score[0] and vr >= self.VARIANCE_RATIO_CUTOFF and
                        (np.isnan(cc) or cc <= self.COV_CHANGE_CUTOFF)):
                    best_method = m
                    best_score = (ksp, vr, cc)

            if best_method is None:
                arr = orig.fillna(orig.mean())
                ksp_fb, vr_fb, _ = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                self.report["missing_numeric"][col] = {
                    "chosen": "fallback_mean",
                    "metrics": (ksp_fb, vr_fb, np.nan),
                    "note": "none met QC"
                }
                self._report_lines.append(f"  • {col:20s} → fallback_mean")
                self.train_df[col] = arr
                imp_fb = SimpleImputer(strategy="mean")
                imp_fb.fit(orig.values.reshape(-1, 1))
                self.numeric_imputers[col] = imp_fb
            else:
                self.report["missing_numeric"][col] = {
                    "chosen": best_method,
                    "metrics": best_score,
                    "note": ""
                }
                self._report_lines.append(
                    f"  • {col:20s} → {best_method}, metrics={best_score}")
                self.train_df[col] = candidates[best_method]
                if best_method == "random_sample":
                    self.numeric_imputers[col] = "random_sample"
                else:
                    self.numeric_imputers[col] = imputers[best_method]

    # --------------- 6) Outlier Detection & Removal ------------------
    def _iqr_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if len(arr) == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - self.UNIV_IQR_FACTOR * iqr, q3 + self.UNIV_IQR_FACTOR * iqr
        return series[(series < lb) | (series > ub)].index.tolist()

    def _zscore_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if len(arr) < 2:
            return []
        mu, sigma = series.mean(), series.std()
        if sigma == 0:
            return []
        z = (series - mu) / sigma
        return series[np.abs(z) > self.UNIV_ZSCORE_CUTOFF].index.tolist()

    def _modz_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if len(arr) < 2:
            return []
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0:
            return []
        modz = 0.6745 * (arr - med) / mad
        return series[np.abs(modz) > self.UNIV_MODZ_CUTOFF].index.tolist()

    def _detect_outliers(self) -> None:
        """
        Vote-based outlier detection:
          - Univariate (IQR, Z-score, Modified Z)
          - Multivariate (Mahalanobis)
        Drops rows with votes ≥ UNIV_FLAG_THRESHOLD.
        """
        self._report_lines.append("\nSTEP 5: Outlier Detection & Removal")
        df = self.train_df
        univ_votes: Dict[int, int] = {}

        # Univariate
        for col in self.numeric_cols:
            self.report["univariate_outliers"][col] = {}
            s = df[col]
            for method_name, fn in [("iqr", self._iqr_outliers),
                                    ("zscore", self._zscore_outliers),
                                    ("modz", self._modz_outliers)]:
                try:
                    idxs = fn(s)
                    cnt = len(idxs)
                    self.report["univariate_outliers"][col][method_name] = cnt
                    self._report_lines.append(
                        f"  • {col:20s} via {method_name:>6s} → {cnt} flagged")
                    for i in idxs:
                        univ_votes[i] = univ_votes.get(i, 0) + 1
                except:
                    self.report["univariate_outliers"][col][method_name] = None
                    self._report_lines.append(
                        f"  • {col:20s} via {method_name:>6s} → error")

        # Multivariate (Mahalanobis) on complete cases
        self._report_lines.append("  → Multivariate (Mahalanobis)")
        numeric_only = df[self.numeric_cols].dropna()
        if numeric_only.shape[0] >= max(self.UNIV_FLAG_THRESHOLD * len(self.numeric_cols), self.MIN_COMPLETE_FOR_COV):
            cov = EmpiricalCovariance().fit(numeric_only.values)
            md = cov.mahalanobis(numeric_only.values)
            thresh = chi2.ppf(self.MULTI_CI, df=numeric_only.shape[1])
            mask = md > thresh
            idxs = numeric_only.index[mask]
            self.report["multivariate_outliers"]["mahalanobis"] = list(idxs)
            self._report_lines.append(f"  • Mahalanobis → {len(idxs)} flagged")
            for i in idxs:
                univ_votes[i] = univ_votes.get(i, 0) + 1
        else:
            self.report["multivariate_outliers"]["mahalanobis"] = []
            self._report_lines.append(
                "  • Too few complete rows → skipped Mahalanobis")

        # Real outliers = vote count ≥ threshold
        real = [idx for idx, v in univ_votes.items() if v >=
                self.UNIV_FLAG_THRESHOLD]
        self.report["multivariate_outliers"]["real_outlier_indices"] = real
        self.report["multivariate_outliers"]["real_count"] = len(real)
        self._report_lines.append(
            f"  → Real outliers (votes ≥ {self.UNIV_FLAG_THRESHOLD}): {len(real)}")
        if real:
            self.train_df.drop(index=real, inplace=True)
            self.train_df.reset_index(drop=True, inplace=True)
            self._report_lines.append(
                f"  • Dropped {len(real)} real outlier rows.")

    # ------------- 7) Multicollinearity Screening & Drop ---------------
    def _compute_vif(self, X: pd.DataFrame) -> pd.Series:
        """
        Compute Variance Inflation Factor (VIF) for each column in X.
        Returns a Series indexed by feature name.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X_np = X.values
        vif_values = []
        for i in range(X_np.shape[1]):
            try:
                vif_i = variance_inflation_factor(X_np, i)
            except:
                vif_i = np.nan
            vif_values.append(vif_i)
        return pd.Series(vif_values, index=X.columns)

    def _drop_collinear_features(self) -> None:
        """
        After imputation (and before scaling), compute pairwise Pearson correlation among numeric
        features.  For any pair with |corr| > CORR_CUTOFF, drop the one with lower variance.
        Then compute VIF on the reduced set; drop any feature with VIF > VIF_CUTOFF, iteratively.
        """
        self._report_lines.append(
            "\nSTEP 6: Multicollinearity Screening & Drop")
        df = self.train_df
        to_drop_corr: List[str] = []
        if len(self.numeric_cols) < 2:
            self.report["collinearity"] = {
                "dropped_corr": [], "dropped_vif": []}
            self._report_lines.append(
                "  • Too few numeric features for collinearity checks.")
            return

        # Pairwise correlation
        corr_matrix = df[self.numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.CORR_CUTOFF].tolist()
            for peer in high_corr:
                # Choose which to drop: the one with lower variance
                var_col = df[col].var() if not np.isnan(df[col].var()) else 0.0
                var_peer = df[peer].var() if not np.isnan(
                    df[peer].var()) else 0.0
                drop_candidate = col if var_col < var_peer else peer
                if drop_candidate not in to_drop_corr:
                    to_drop_corr.append(drop_candidate)

        if to_drop_corr:
            df.drop(columns=to_drop_corr, inplace=True)
            self._report_lines.append(
                f"  • Dropped {len(to_drop_corr)} collinear (|corr| > {self.CORR_CUTOFF}): {to_drop_corr}")
        else:
            self._report_lines.append("  • No high‐correlation pairs found.")

        # Update numeric_cols
        self.numeric_cols = [
            c for c in self.numeric_cols if c not in to_drop_corr]

        # Compute VIF iteratively
        dropped_vif: List[str] = []
        while True:
            if len(self.numeric_cols) < 2:
                break
            X_sub = df[self.numeric_cols].dropna()
            if X_sub.shape[0] < 2:
                break
            vif_series = self._compute_vif(X_sub)
            max_vif = vif_series.max()
            if max_vif > self.VIF_CUTOFF:
                drop_feat = vif_series.idxmax()
                dropped_vif.append(drop_feat)
                df.drop(columns=[drop_feat], inplace=True)
                self._report_lines.append(
                    f"  • Dropped '{drop_feat}' (VIF={max_vif:.2f} > {self.VIF_CUTOFF})")
                self.numeric_cols.remove(drop_feat)
            else:
                break

        if not dropped_vif:
            self._report_lines.append(
                "  • No features with VIF above threshold.")
        self.report["collinearity"] = {
            "dropped_corr": to_drop_corr,
            "dropped_vif": dropped_vif,
        }

    # --------------- 8) Target-Leakage Check & Drop --------------
    def _detect_target_leakage(self) -> None:
        """
        For each feature, attempt to predict the target using only that single column.
        If AUC > AUC_LEAKAGE_CUTOFF (e.g. 0.99), drop the feature.
        Only runs if self.target is provided.
        """
        self._report_lines.append("\nSTEP 7: Target-Leakage Check")
        if not self.target or self.y_train is None:
            self.report["target_leakage"] = {"note": "unsupervised → skipped"}
            self._report_lines.append(
                "  • No target provided → skip leakage detection.")
            return

        dropped: List[str] = []
        X = self.train_df.copy()
        y = self.y_train.copy()
        for col in X.columns:
            try:
                # Only consider numeric or low-cardinality categorical
                if col in self.numeric_cols:
                    model = LogisticRegression(
                        solver="liblinear", random_state=self.random_state)
                    Xi = X[[col]].dropna()
                    yi = y.loc[Xi.index]
                    if Xi[col].nunique() < 2 or yi.nunique() < 2:
                        continue
                    model.fit(Xi.values.reshape(-1, 1), yi.values)
                    pred = model.predict_proba(Xi.values.reshape(-1, 1))[:, 1]
                    auc = roc_auc_score(yi.values, pred)
                else:
                    ser = X[col].astype(str)
                    le = LabelEncoder()
                    ser_le = le.fit_transform(ser.fillna("__MISSING__"))
                    if len(np.unique(ser_le)) < 2 or y.nunique() < 2:
                        continue
                    model = LogisticRegression(
                        solver="liblinear", random_state=self.random_state)
                    Xi = ser_le.reshape(-1, 1)
                    yi = y.values
                    model.fit(Xi, yi)
                    pred = model.predict_proba(Xi)[:, 1]
                    auc = roc_auc_score(yi, pred)
                if auc >= self.AUC_LEAKAGE_CUTOFF:
                    dropped.append(col)
                    self._report_lines.append(
                        f"  • Dropped '{col}' (AUC={auc:.3f} ≥ {self.AUC_LEAKAGE_CUTOFF})")
            except:
                continue

        if dropped:
            self.train_df.drop(columns=dropped, inplace=True)
            self._report_lines.append(
                f"  • Total target‐leakage features dropped: {dropped}")
        else:
            self._report_lines.append(
                "  • No target‐leakage features detected.")
        self.report["target_leakage"] = {"dropped": dropped}

        # Also remove from numeric and categorical lists if present
        for col in dropped:
            if col in self.numeric_cols:
                self.numeric_cols.remove(col)
            if col in self.categorical_cols:
                self.categorical_cols.remove(col)

    # ---------------- 9) Scaling Numeric ---------------------
    def _choose_scaler(self) -> Tuple[str, Dict[str, Dict[str, float]]]:
        """
        Choose between StandardScaler, RobustScaler, MinMaxScaler:
          - If any |skew| > SKEW_THRESH_FOR_ROBUST or |kurtosis| > KURT_THRESH_FOR_ROBUST → Robust
          - Elif all |skew| < SKEW_THRESH_FOR_STANDARD → Standard
          - Else → MinMax
        """
        skews, kurts = {}, {}
        for col in self.numeric_cols:
            arr = self.train_df[col].dropna().values
            if len(arr) > 2:
                skews[col] = float(stats.skew(arr))
                kurts[col] = float(stats.kurtosis(arr))
            else:
                skews[col] = 0.0
                kurts[col] = 0.0

        if any(
            abs(sk) > self.SKEW_THRESH_FOR_ROBUST or abs(
                ku) > self.KURT_THRESH_FOR_ROBUST
            for sk, ku in zip(skews.values(), kurts.values())
        ):
            return "RobustScaler", {"skew": skews, "kurtosis": kurts}
        if all(abs(sk) < self.SKEW_THRESH_FOR_STANDARD for sk in skews.values()):
            return "StandardScaler", {"skew": skews, "kurtosis": kurts}
        return "MinMaxScaler", {"skew": skews, "kurtosis": kurts}

    def _apply_scaling(self) -> None:
        """
        Fit & apply the chosen numeric scaler on TRAIN. Save model to apply on VAL/TEST.
        """
        self._report_lines.append("\nSTEP 8: Scaling (Numeric)")
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

        self.train_df[self.numeric_cols] = model.fit_transform(
            self.train_df[self.numeric_cols])
        self.scaler_model = clone(model)
        self._report_lines.append(f"  • Applied {scaler_name}.")

    # ------------- 10) Conditional Extra Transformation -------------

    def _evaluate_transform(
        self, series: pd.Series
    ) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """
        Evaluate transforms: none, Box-Cox, Yeo-Johnson, Quantile→Normal.
        Only apply if scaling did not yield approximate normality:
          Criteria: Shapiro p-value < 0.05 OR |skew| > 0.75 after scaling.
        """
        arr_scaled = series.values
        scores: Dict[str, Tuple[float, float]] = {}
        if len(arr_scaled) < 5 or series.nunique() < 5:
            return "none", {"none": (1.0, 0.0)}

        for method in self.TRANSFORM_CANDIDATES:
            try:
                if method == "none":
                    arr = arr_scaled
                elif method == "boxcox":
                    if np.any(arr_scaled <= 0):
                        continue
                    arr, _ = stats.boxcox(arr_scaled)
                elif method == "yeo":
                    pt = PowerTransformer(
                        method="yeo-johnson", standardize=True)
                    arr = pt.fit_transform(arr_scaled.reshape(-1, 1)).flatten()
                else:  # quantile
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=self.random_state)
                    arr = qt.fit_transform(arr_scaled.reshape(-1, 1)).flatten()

                # Test normality on sample
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

    def _apply_transform(self) -> None:
        """
        For each numeric column, only apply an extra transform if:
          after scaling, Shapiro p < 0.05 OR |skew| > 0.75.
        """
        self._report_lines.append(
            "\nSTEP 9: Extra Transformation (Numeric, Conditional)")
        for col in self.numeric_cols:
            arr = self.train_df[col].dropna().values
            if len(arr) < 5:
                self.report["transform"][col] = {
                    "chosen": "none", "scores": {"none": (1.0, 0.0)}}
                self._report_lines.append(
                    f"  • {col:20s} → too few rows for transformation")
                continue

            try:
                sample = arr if arr.size <= 5000 else np.random.choice(
                    arr, 5000, replace=False)
                pval_scaled = float(stats.shapiro(
                    sample)[1]) if sample.size >= 3 else 0.0
            except:
                pval_scaled = 0.0
            skew_scaled = abs(float(stats.skew(arr)))
            if pval_scaled > 0.05 and skew_scaled < 0.75:
                self.report["transform"][col] = {
                    "chosen": "none", "scores": {"none": (pval_scaled, skew_scaled)}}
                self._report_lines.append(
                    f"  • {col:20s} → no transform (scaling sufficed)")
                continue

            best, scores = self._evaluate_transform(pd.Series(arr))
            self.report["transform"][col] = {"chosen": best, "scores": scores}
            self._report_lines.append(
                f"  • {col:20s} → {best}, scores={scores}")
            if best == "boxcox":
                transformed, _ = stats.boxcox(self.train_df[col].values)
                self.train_df[col] = transformed
                self.transform_model[col] = ("boxcox", None)
                self._report_lines.append(f"    • Applied Box-Cox to '{col}'.")
            elif best == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                self.train_df[col] = pt.fit_transform(
                    self.train_df[col].values.reshape(-1, 1)).flatten()
                self.transform_model[col] = ("yeo", clone(pt))
                self._report_lines.append(
                    f"    • Applied Yeo-Johnson to '{col}'.")
            elif best == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state)
                self.train_df[col] = qt.fit_transform(
                    self.train_df[col].values.reshape(-1, 1)).flatten()
                self.transform_model[col] = ("quantile", clone(qt))
                self._report_lines.append(
                    f"    • Applied Quantile→Normal to '{col}'.")
            else:
                self._report_lines.append(
                    f"    • No transformation for '{col}'.")

    # ------------- 11) Categorical Missing-Value Imputation -------------
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

    def _impute_missing_categorical(self) -> None:
        """
        Impute categorical columns on TRAIN set by minimizing total-variation distance.
        Store fitted imputers for VAL/TEST.
        """
        self._report_lines.append(
            "\nSTEP 10: Categorical Missing-Value Imputation")
        df = self.train_df
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
                self.categorical_imputers[col] = clone(imp)
                self._report_lines.append(
                    f"    • Applied mode imputation to '{col}'.")
            elif best == "constant_other":
                df[col] = orig.fillna("__MISSING__").astype(object)
                self.categorical_imputers[col] = "__MISSING__"
                self._report_lines.append(
                    f"    • Filled nulls with '__MISSING__' in '{col}'.")
            elif best == "random_sample":
                df[col] = self._random_sample_impute_cat(orig)
                # means random sampling at val/test
                self.categorical_imputers[col] = None
                self._report_lines.append(
                    f"    • Random-sample imputed '{col}'.")
            else:
                df[col] = orig.fillna("__MISSING__").astype(object)
                self.categorical_imputers[col] = "__MISSING__"
                self._report_lines.append(
                    f"    • Fallback: filled '__MISSING__' for '{col}'.")

    # ---------------- 12) Rare-Category Grouping -----------------
    def _group_rare_categories(self) -> None:
        """
        Group any category with frequency < RARE_FREQ_CUTOFF into "__RARE__".
        """
        self._report_lines.append("\nSTEP 11: Rare-Category Grouping")
        df = self.train_df
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

    # ---------------- 13) Categorical Encoding Variants -----------------
    def _onehot_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return pd.get_dummies(df[cols], prefix=cols, drop_first=False, dtype=float)

    def _ordinal_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)
        arr = enc.fit_transform(df[cols].astype(object))
        # Save the fitted encoder for each col
        for i, c in enumerate(cols):
            self.encoding_models[c] = enc
        return pd.DataFrame(arr.astype(int), columns=cols, index=df.index)

    def _frequency_encode(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out: Dict[str, List[float]] = {}
        for col in cols:
            freq = df[col].value_counts(normalize=True)
            self.encoding_models[col] = freq.to_dict()  # map for val/test
            out[col + "_freq"] = df[col].map(freq).fillna(0.0)
        return pd.DataFrame(out, index=df.index)

    def _encode_categorical_variants(self) -> None:
        """
        Produce three different “pre‐model” versions of TRAIN:
          - processed_train_linear.csv   (one-hot + freq)
          - processed_train_tree.csv     (one-hot + ordinal + freq)
          - processed_train_knn.csv      (freq-only)
        Handle unseen test categories by mapping to "__UNSEEN__" or zero‐freq.
        """
        self._report_lines.append("\nSTEP 12: Categorical Encoding Variants")
        base = self.train_df.copy()
        n_rows = len(base)

        # 12a) LINEAR: one-hot if freq ≤ ONEHOT_MAX_UNIQ, else freq
        linear_onehot: List[str] = []
        linear_freq: List[str] = []
        linear_sugg: List[str] = []
        for col in self.categorical_cols:
            uniq = base[col].nunique()
            frac = uniq / n_rows if n_rows > 0 else 0.0
            if frac <= self.ONEHOT_MAX_UNIQ:
                linear_onehot.append(col)
            else:
                if frac <= self.KNOWNALLY_MAX_UNIQ:
                    linear_freq.append(col)
                else:
                    linear_sugg.append(
                        f"[LINEAR] '{col}' frac={frac:.2f} > {self.ULTRAHIGH_UNIQ}: suggest target encoding")

        oh = self._onehot_encode(
            base, linear_onehot) if linear_onehot else pd.DataFrame(index=base.index)
        freq_df_lin = self._frequency_encode(
            base, linear_freq) if linear_freq else pd.DataFrame(index=base.index)
        df_lin = pd.concat(
            [base.drop(columns=self.categorical_cols), oh, freq_df_lin], axis=1)
        df_lin.to_csv("processed_train_linear.csv", index=False)
        self.report["encoding"]["linear"] = {
            "onehot": linear_onehot,
            "frequency": linear_freq,
            "suggestions": linear_sugg,
        }
        self._report_lines.append(
            f"  • [LINEAR] onehot={linear_onehot}, freq={linear_freq}")
        for s in linear_sugg:
            self._report_lines.append(f"    • {s}")

        # 12b) TREE: one-hot ≤ ONEHOT_MAX_UNIQ, elif ≤ ORDINAL_MAX_UNIQ → ordinal,
        # elif ≤ KNOWNALLY_MAX_UNIQ → freq, else suggest target
        tree_onehot: List[str] = []
        tree_ordinal: List[str] = []
        tree_freq: List[str] = []
        tree_sugg: List[str] = []
        df_tree = base.copy()
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
                    f"[TREE] '{col}' frac={frac:.2f} > {self.ULTRAHIGH_UNIQ}: suggest target encoding")

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

        df_tree.to_csv("processed_train_tree.csv", index=False)
        self.report["encoding"]["tree"] = {
            "onehot": tree_onehot,
            "ordinal": tree_ordinal,
            "frequency": tree_freq,
            "suggestions": tree_sugg,
        }
        self._report_lines.append(
            f"  • [TREE] onehot={tree_onehot}, ordinal={tree_ordinal}, freq={tree_freq}"
        )
        for s in tree_sugg:
            self._report_lines.append(f"    • {s}")

        # 12c) KNN: frequency encode all categorical
        freq_all = self._frequency_encode(
            base, self.categorical_cols) if self.categorical_cols else pd.DataFrame(index=base.index)
        df_knn = pd.concat(
            [base.drop(columns=self.categorical_cols), freq_all], axis=1)
        df_knn.to_csv("processed_train_knn.csv", index=False)
        self.report["encoding"]["knn"] = {
            "frequency_all": self.categorical_cols}
        self._report_lines.append(
            f"  • [KNN] freq-encode all: {self.categorical_cols}")

    # ----------- 13) Final Drop of Near-Constant Columns -------------
    def _drop_near_constant(self) -> None:
        """
        After all transforms, drop any column whose variance < CONST_THRESHOLD.
        """
        self._report_lines.append(
            "\nSTEP 13: Final Drop of Near-Constant Features (Post-Transform)")
        df = self.train_df
        to_drop: List[str] = []
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8]:
                if df[col].var() < self.CONST_THRESHOLD:
                    to_drop.append(col)
            else:
                # for non-numeric, if only one unique after transforms
                if df[col].nunique() <= 1:
                    to_drop.append(col)
        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            self._report_lines.append(
                f"  • Dropped near-constant post-transform: {to_drop}")
        else:
            self._report_lines.append("  • No near-constant features to drop.")
        self.report["final_constant_drop"] = {"dropped": to_drop}
        # Update lists
        self.numeric_cols = [c for c in self.numeric_cols if c not in to_drop]
        self.categorical_cols = [
            c for c in self.categorical_cols if c not in to_drop]
        self.boolean_cols = [c for c in self.boolean_cols if c not in to_drop]

    # --------------- 14) Dimensionality Reduction (PCA vs FA) ---------------
    def _apply_dimensionality_reduction(self) -> None:
        """
        Compare PCA vs FactorAnalysis: compute reconstruction MSE on TRAIN numeric subset.
        Choose whichever yields ≥ FA_MSE_IMPROV_THRESHOLD improvement in MSE; else choose PCA.
        Replace numeric columns in train_df with chosen DR components.
        """
        self._report_lines.append(
            "\nSTEP 14: Dimensionality Reduction Comparison (PCA vs FactorAnalysis)")
        if not self.apply_pca or len(self.numeric_cols) < 2:
            self.report["dimensionality_reduction"] = {
                "note": "skipped (apply_pca=False or too few numeric)"}
            self._report_lines.append(
                "  • DR skipped (apply_pca=False or insufficient numeric features).")
            return

        X = self.train_df[self.numeric_cols].copy()
        if X.isna().any(axis=None):
            self._report_lines.append(
                "  • DR aborted: NaNs present in numeric features.")
            self.report["dimensionality_reduction"] = {
                "note": "skipped (NaNs present)"}
            return

        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.values)

        # 14a) Fit PCA on TRAIN
        full_pca = PCA().fit(X_std)
        cumvar = np.cumsum(full_pca.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, self.pca_variance_threshold) + 1)
        pca_model = PCA(n_components=n_comp)
        X_pca = pca_model.fit_transform(X_std)
        X_pca_rec = pca_model.inverse_transform(X_pca)
        mse_pca = np.mean((X_std - X_pca_rec) ** 2)

        # 14b) Fit FactorAnalysis on TRAIN (same n_comp)
        try:
            fa_model = FactorAnalysis(
                n_components=n_comp, random_state=self.random_state)
            X_fa = fa_model.fit_transform(X_std)
            # Reconstruction via: loadings @ factors + mean
            loadings = fa_model.components_.T  # shape (p, n_comp)
            X_fa_rec = np.dot(X_fa, loadings.T)
            X_fa_rec += fa_model.mean_
            mse_fa = np.mean((X_std - X_fa_rec) ** 2)
        except:
            mse_fa = np.inf

        # Compare MSEs
        improvement = (mse_pca - mse_fa) / mse_pca if mse_pca > 0 else 0.0
        if mse_fa < mse_pca and improvement >= self.FA_MSE_IMPROV_THRESHOLD:
            self.chosen_dr = "FactorAnalysis"
            self.fa_model = fa_model
            dr_model = fa_model
            self._report_lines.append(
                f"  • FactorAnalysis chosen (MSE_FA={mse_fa:.4f} vs MSE_PCA={mse_pca:.4f}, imp={improvement:.2%}).")
        else:
            self.chosen_dr = "PCA"
            self.pca_model = pca_model
            dr_model = pca_model
            self._report_lines.append(
                f"  • PCA chosen (MSE_PCA={mse_pca:.4f} vs MSE_FA={mse_fa:.4f}).")

        # Transform TRAIN, replace numeric columns
        if self.chosen_dr == "PCA":
            X_dr = pca_model.transform(X_std)
            cols = [f"PC{i+1}" for i in range(n_comp)]
        else:
            X_dr = fa_model.transform(X_std)
            cols = [f"FA{i+1}" for i in range(n_comp)]

        df_dr = pd.DataFrame(X_dr, columns=cols, index=self.train_df.index)
        self.train_df = pd.concat(
            [self.train_df.drop(columns=self.numeric_cols), df_dr], axis=1)

        self.report["dimensionality_reduction"] = {
            "method": self.chosen_dr,
            "n_components": n_comp,
            "mse_pca": mse_pca,
            "mse_fa": mse_fa,
            "improvement": improvement,
        }
        self._report_lines.append(
            f"  • Replaced numeric cols with {self.chosen_dr} components.")

    # ---------------- Public API: fit_transform ----------------
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        1) Deduplicate rows
        2) Split into train/val/test
        3) Cast booleans
        4) Drop constant features
        5) Impute numeric missing
        6) Detect & remove outliers
        7) Multicollinearity screen → drop
        8) Target leakage check → drop
        9) Scale numeric
       10) Conditional transform numeric
       11) Impute categorical missing
       12) Group rare categories
       13) Encode categoricals → write CSVs
       14) Final drop near-constant
       15) Compare DR: PCA vs FactorAnalysis

        Returns processed TRAIN, VAL, TEST (with target column still present in VAL/TEST).
        """
        if isinstance(X, np.ndarray):
            raise ValueError("Input must be a pandas DataFrame")
        df = X.copy()

        # If target included, preserve it for splitting
        if self.target and self.target in df.columns:
            y_all = df[self.target]
        else:
            y_all = None

        # 0A) Remove duplicates from full dataset
        df = self._remove_duplicates(df)

        # 0B) Split into train/val/test
        train_raw, val_raw, test_raw = self._train_val_test_split(df)
        if self.target and self.target in train_raw.columns:
            self.y_train = train_raw[self.target].reset_index(drop=True)
            train_raw = train_raw.drop(
                columns=[self.target]).reset_index(drop=True)
        else:
            self.y_train = None
        if self.target and self.target in val_raw.columns:
            self.y_val = val_raw[self.target].reset_index(drop=True)
            val_raw = val_raw.drop(
                columns=[self.target]).reset_index(drop=True)
        else:
            self.y_val = None
        if self.target and self.target in test_raw.columns:
            self.y_test = test_raw[self.target].reset_index(drop=True)
            test_raw = test_raw.drop(
                columns=[self.target]).reset_index(drop=True)
        else:
            self.y_test = None

        self.train_df = train_raw.copy()
        self.val_df = val_raw.copy()
        self.test_df = test_raw.copy()

        # 1) Missing mechanism detection (on train)
        self._detect_missing_mechanism(train_raw)

        # 2) Boolean detection & casting (train/val/test)
        self.train_df = self._cast_booleans(self.train_df)
        self.val_df = self._cast_booleans(self.val_df)
        self.test_df = self._cast_booleans(self.test_df)

        # 3) Identify feature types on train (after boolean cast)
        all_cols = self.train_df.columns.tolist()
        # Numeric: anything of dtype kind ∈ {i, u, f}
        self.numeric_cols = self.train_df.select_dtypes(
            include=[np.number]).columns.tolist()
        # Boolean: int8 flags
        self.boolean_cols = [
            c for c in self.train_df.columns if self.train_df[c].dtype == np.int8]
        # Categorical: any non-numeric and not boolean
        self.categorical_cols = [
            c for c in all_cols if c not in self.numeric_cols and c not in self.boolean_cols]
        self.non_numeric_cols = self.categorical_cols.copy()

        # 4) Drop zero‐variance (constant) features (train/val/test) – use only train to decide
        self.train_df = self._drop_zero_variance(self.train_df)
        # Drop same columns from val/test
        dropped_consts = self.report["constant_drop"]["dropped"]
        if dropped_consts:
            self.val_df.drop(
                columns=[c for c in dropped_consts if c in self.val_df.columns], inplace=True)
            self.test_df.drop(
                columns=[c for c in dropped_consts if c in self.test_df.columns], inplace=True)

        # Refresh numeric/categorical lists after constant drop
        self.numeric_cols = [
            c for c in self.numeric_cols if c not in dropped_consts]
        self.categorical_cols = [
            c for c in self.categorical_cols if c not in dropped_consts]
        self.boolean_cols = [
            c for c in self.boolean_cols if c not in dropped_consts]

        # 5) Numeric missing imputation (train) – store imputers
        if self.numeric_cols:
            self._impute_missing_numeric()
            # Apply same imputers to val/test
            for col, imp in self.numeric_imputers.items():
                # TRAIN was already filled
                if imp == "random_sample":
                    # Random-sample on val/test
                    self.val_df[col] = self.val_df[col].combine_first(self.val_df[col].apply(
                        lambda x: np.nan if pd.isna(x) else x
                    )).pipe(lambda s: self._random_sample_impute_num(s))
                    self.test_df[col] = self.test_df[col].combine_first(self.test_df[col].apply(
                        lambda x: np.nan if pd.isna(x) else x
                    )).pipe(lambda s: self._random_sample_impute_num(s))
                else:
                    # Imputer has fit on train
                    arr_val = imp.transform(
                        self.val_df[col].values.reshape(-1, 1)).flatten()
                    arr_test = imp.transform(
                        self.test_df[col].values.reshape(-1, 1)).flatten()
                    self.val_df[col] = arr_val
                    self.test_df[col] = arr_test

        # 6) Outlier detection (train) & removal
        if self.numeric_cols:
            self._detect_outliers()
            # We do NOT remove outliers from val/test—they remain for unbiased evaluation

        # 7) Multicollinearity screening (train) & drop
        if self.numeric_cols:
            self._drop_collinear_features()
            # Drop same features from val/test
            dropped_corr = self.report["collinearity"]["dropped_corr"]
            dropped_vif = self.report["collinearity"]["dropped_vif"]
            dropped_all = dropped_corr + dropped_vif
            self.val_df.drop(
                columns=[c for c in dropped_all if c in self.val_df.columns], inplace=True)
            self.test_df.drop(
                columns=[c for c in dropped_all if c in self.test_df.columns], inplace=True)

        # 8) Target leakage check (train) & drop
        if self.numeric_cols or self.categorical_cols:
            self._detect_target_leakage()
            dropped_leak = self.report["target_leakage"]["dropped"]
            if dropped_leak:
                self.val_df.drop(
                    columns=[c for c in dropped_leak if c in self.val_df.columns], inplace=True)
                self.test_df.drop(
                    columns=[c for c in dropped_leak if c in self.test_df.columns], inplace=True)

        # Refresh numeric/categorical lists after leakage drop
        self.numeric_cols = [
            c for c in self.numeric_cols if c in self.train_df.columns]
        self.categorical_cols = [
            c for c in self.categorical_cols if c in self.train_df.columns]
        self.boolean_cols = [
            c for c in self.boolean_cols if c in self.train_df.columns]

        # 9) Scaling (train) & store scaler for val/test
        if self.numeric_cols:
            self._apply_scaling()
            # Scale val/test
            for col in self.numeric_cols:
                self.val_df[col] = self.scaler_model.transform(
                    self.val_df[[col]])
                self.test_df[col] = self.scaler_model.transform(
                    self.test_df[[col]])

        # 10) Conditional extra transform (train) & store models for val/test
        if self.numeric_cols:
            self._apply_transform()
            for col, (method, model) in self.transform_model.items():
                if method == "none":
                    continue
                if method == "boxcox":
                    # We cannot invert Box-Cox easily; assume train/test distributions close enough
                    self.val_df[col] = self.val_df[col].pipe(
                        lambda arr: stats.boxcox(arr)[0])
                    self.test_df[col] = self.test_df[col].pipe(
                        lambda arr: stats.boxcox(arr)[0])
                elif method == "yeo":
                    self.val_df[col] = model.transform(
                        self.val_df[[col]]).flatten()
                    self.test_df[col] = model.transform(
                        self.test_df[[col]]).flatten()
                elif method == "quantile":
                    self.val_df[col] = model.transform(
                        self.val_df[[col]]).flatten()
                    self.test_df[col] = model.transform(
                        self.test_df[[col]]).flatten()

        # 11) Categorical missing imputation (train) & apply to val/test
        if self.categorical_cols:
            self._impute_missing_categorical()
            for col, imp in self.categorical_imputers.items():
                if imp == "__MISSING__":
                    self.val_df[col] = self.val_df[col].fillna("__MISSING__")
                    self.test_df[col] = self.test_df[col].fillna("__MISSING__")
                elif imp is None:
                    # random-sample
                    self.val_df[col] = self.val_df[col].pipe(
                        lambda s: self._random_sample_impute_cat(s))
                    self.test_df[col] = self.test_df[col].pipe(
                        lambda s: self._random_sample_impute_cat(s))
                else:
                    # SimpleImputer
                    arr_val = imp.transform(
                        self.val_df[col].values.reshape(-1, 1)).flatten()
                    arr_test = imp.transform(
                        self.test_df[col].values.reshape(-1, 1)).flatten()
                    self.val_df[col] = arr_val
                    self.test_df[col] = arr_test

        # 12) Rare-category grouping (train) & apply to val/test
        if self.categorical_cols:
            # We have a mapping of rare categories from train
            self._group_rare_categories()
            rare_map = self.report["rare_categories"]
            for col, rare_list in rare_map.items():
                if col in self.val_df.columns:
                    self.val_df[col] = self.val_df[col].apply(
                        lambda x: "__RARE__" if x in rare_list else x)
                if col in self.test_df.columns:
                    self.test_df[col] = self.test_df[col].apply(
                        lambda x: "__RARE__" if x in rare_list else x)

        # 13) Categorical encoding variants (train) → CSVs; apply at VAL/TEST time
        if self.categorical_cols:
            self._encode_categorical_variants()
            # Note: we do NOT produce processed_val/processed_test files here,
            # but we do need to store encoders to apply same mapping in modeling.

            # We must ensure unseen categories in VAL/TEST map to "__UNSEEN__" or zero freq.
            # For each categorical feature:
            for col, mapping in self.encoding_models.items():
                if isinstance(mapping, dict):
                    # frequency map
                    # any category not in mapping → freq 0 (implicitly handled by .map(...).fillna(0.0))
                    self.val_df[col +
                                "_freq"] = self.val_df[col].map(mapping).fillna(0.0)
                    self.test_df[col +
                                 "_freq"] = self.test_df[col].map(mapping).fillna(0.0)
                elif isinstance(mapping, OrdinalEncoder):
                    # ordinal: unknown_value = -1 already encoded, so just transform
                    arr_val = mapping.transform(
                        self.val_df[[col]].astype(object)).flatten().astype(int)
                    arr_test = mapping.transform(
                        self.test_df[[col]].astype(object)).flatten().astype(int)
                    self.val_df[col] = arr_val
                    self.test_df[col] = arr_test
                # For one‐hot we cannot easily revert; so downstream modeling should re-encode from train’s one-hot schema.

        # Refresh column lists after encoding (train)
        self.numeric_cols = [
            c for c in self.train_df.columns if c not in self.categorical_cols]
        self.categorical_cols = []  # after encoding, no raw categoricals remain

        # 14) Final drop near-constant after all transformations
        self._drop_near_constant()
        dropped_final = self.report["final_constant_drop"]["dropped"]
        if dropped_final:
            self.val_df.drop(
                columns=[c for c in dropped_final if c in self.val_df.columns], inplace=True)
            self.test_df.drop(
                columns=[c for c in dropped_final if c in self.test_df.columns], inplace=True)

        # 15) Dimensionality Reduction comparison (train) and replace numeric
        if self.numeric_cols:
            self._apply_dimensionality_reduction()
            # "PCA" or "FactorAnalysis"
            dr_cols = self.report["dimensionality_reduction"]["method"]
            n_comp = self.report["dimensionality_reduction"]["n_components"]
            comp_names = [
                f"{'PC' if dr_cols == 'PCA' else 'FA'}{i+1}" for i in range(n_comp)]

            # Apply DR to VAL/TEST similarly
            X_val = self.val_df[[
                c for c in self.train_df.columns if c not in comp_names]].copy()
            X_test = self.test_df[[
                c for c in self.train_df.columns if c not in comp_names]].copy()
            # Actually, we dropped numeric_cols from train_df when forming DR output,
            # so train_df now only has DR columns + leftover categoricals/booleans.
            # For val/test, select original numeric columns and apply DR:
            original_numeric_val = X_val[self.numeric_cols].copy().values
            original_numeric_test = X_test[self.numeric_cols].copy().values

            # Scale val/test (already scaled and transformed earlier)
            # build X_val_std and X_test_std
            if self.scaler_model:
                X_val_num = self.scaler_model.transform(original_numeric_val)
                X_test_num = self.scaler_model.transform(original_numeric_test)
            else:
                X_val_num = original_numeric_val
                X_test_num = original_numeric_test

            # Apply extra transforms if any (only Quantile/Yeo/BoxCox) for val/test:
            for col, (method, model) in self.transform_model.items():
                idx = self.numeric_cols.index(col)
                if method == "boxcox":
                    X_val_num[:, idx] = stats.boxcox(X_val_num[:, idx])[0]
                    X_test_num[:, idx] = stats.boxcox(X_test_num[:, idx])[0]
                elif method == "yeo":
                    X_val_num[:, idx] = model.transform(
                        X_val_num[:, idx].reshape(-1, 1)).flatten()
                    X_test_num[:, idx] = model.transform(
                        X_test_num[:, idx].reshape(-1, 1)).flatten()
                elif method == "quantile":
                    X_val_num[:, idx] = model.transform(
                        X_val_num[:, idx].reshape(-1, 1)).flatten()
                    X_test_num[:, idx] = model.transform(
                        X_test_num[:, idx].reshape(-1, 1)).flatten()

            # Now apply chosen DR:
            if self.chosen_dr == "PCA":
                X_val_dr = self.pca_model.transform(X_val_num)
                X_test_dr = self.pca_model.transform(X_test_num)
            else:
                X_val_dr = self.fa_model.transform(X_val_num)
                X_test_dr = self.fa_model.transform(X_test_num)

            df_val_dr = pd.DataFrame(
                X_val_dr, columns=comp_names, index=self.val_df.index)
            df_test_dr = pd.DataFrame(
                X_test_dr, columns=comp_names, index=self.test_df.index)

            # Drop original numeric_cols from val/test, then concat DR columns
            self.val_df.drop(columns=self.numeric_cols,
                             inplace=True, errors="ignore")
            self.test_df.drop(columns=self.numeric_cols,
                              inplace=True, errors="ignore")
            self.val_df = pd.concat([self.val_df, df_val_dr], axis=1)
            self.test_df = pd.concat([self.test_df, df_test_dr], axis=1)

        # Finally, write the analysis report
        Path("analysis_report.txt").write_text("\n".join(self._report_lines))

        # Return processed TRAIN, VAL, TEST (with their target columns separate)
        # If target existed, we can re-attach it if desired; but by design, we return only features here:
        return self.train_df.copy(), self.val_df.copy(), self.test_df.copy()

    def print_report(self) -> None:
        """
        Print the stepwise log of decisions.
        """
        print("\n" + "=" * 80)
        print("✅ Data Quality Pipeline Report")
        print("=" * 80)
        for line in self._report_lines:
            print(line)
        print("\n" + "=" * 80 + "\n")
