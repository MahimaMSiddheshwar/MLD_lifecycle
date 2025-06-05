#!/usr/bin/env python3
"""
MLDLC End-to-End Data Quality & Preprocessing Pipeline (Improved)
=================================================================

This version integrates:

  • Train/Test split as first step, with imbalance diagnostics
  • Scalability checks: skip KNN/MICE if dataset too large
  • “Winsorizing” option for univariate outliers; row‐drop only if repeated flags
  • Per‐feature “lifetime” log: every transform/drop is recorded
  • Before/after distribution metrics (skew, kurtosis, Shapiro p)
  • Shannon entropy + mutual information (or χ²) vs. target (“feature screening”)
  • Missingness pattern analysis: Little’s test + target‐missing correlation + missing‐flags
  • Ordinal vs nominal categorical distinction (explicit `ordinal_cols` parameter)
  • Validation of scaler/transform on VAL/TEST: detect out‐of‐range issues
  • Dual output DataFrames for “linear” (one‐hot + freq) vs “tree” encodings
  • Interaction suggestions: top correlated numeric–numeric & numeric–categorical pairs
  • LeakageDetection: checks for target leakage (AUC ≈ 1) or train/test leakage
  • Mixed‐dtype detection for each column

Usage:
    from data_quality import DataQualityPipeline
    import pandas as pd

    df = pd.read_csv("data/raw/your_data.csv")
    dq = DataQualityPipeline(
        target_column="is_churn",
        ordinal_cols=["Low", "Medium", "High"],    # any truly ordinal columns
        test_size=0.2,
        random_state=42,
        pca_variance_threshold=0.90,
        apply_pca=True,
        verbose=True,
        max_impute_small=5000,                    # skip KNN/MICE if n_train > 5000
        outlier_winsorize=True                     # winsorize outliers rather than full row drop
    )
    train_lin, train_tree, val_df = dq.fit_transform(df)
    dq.print_report()
    # Two processed train DataFrames (linear‐encoded & tree‐encoded), plus raw val_df
    # `analysis_report.txt` is written, as is `processed_train_linear.csv` / `processed_train_tree.csv`.
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
from statsmodels.stats.missing import test_missingness

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, OrdinalEncoder
)
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class LeakageDetection:
    """
    Simple utilities to detect:
      1) Target leakage: any feature whose one-vs-rest AUC > threshold.
      2) Train/Test leakage: if a hashing or fingerprint of features
         nearly perfectly separates train vs test.
    """

    def __init__(self, auc_threshold: float = 0.99, verbose: bool = True):
        self.auc_threshold = auc_threshold
        self.verbose = verbose
        self.leakage_report: Dict[str, Dict] = {}

    def check_target_leakage(
        self, df: pd.DataFrame, target: pd.Series, categorical_cols: List[str], numeric_cols: List[str]
    ) -> None:
        """
        For each feature, compute AUC(classifier=feature → target). If AUC > threshold,
        flag as potential leakage.
        """
        leak_feats = []
        self.leakage_report["target_leakage"] = {}
        for col in numeric_cols + categorical_cols:
            try:
                x = df[col]
                # if categorical, convert to one-hot and compute per dummy; we take max across dummies
                if col in categorical_cols:
                    dummies = pd.get_dummies(x, prefix=col, dummy_na=False)
                    aucs = []
                    for dummy_col in dummies.columns:
                        try:
                            auc_val = roc_auc_score(target, dummies[dummy_col])
                            aucs.append(auc_val)
                        except:
                            pass
                    if not aucs:
                        continue
                    max_auc = max(aucs)
                else:
                    # numeric: scale to [0,1], handle constant
                    if x.nunique() <= 1:
                        continue
                    xv = (x - x.min()) / (x.max() - x.min() + 1e-9)
                    max_auc = roc_auc_score(target, xv.fillna(0.5))
                self.leakage_report["target_leakage"][col] = float(max_auc)
                if max_auc >= self.auc_threshold:
                    leak_feats.append(col)
                    if self.verbose:
                        print(
                            f"[LeakageDetection] ⚠ Feature '{col}' AUC={max_auc:.3f} ≥ {self.auc_threshold}")
            except Exception:
                continue
        self.leakage_report["leaky_features"] = leak_feats

    def check_train_test_separation(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]
    ) -> None:
        """
        Check if any feature nearly perfectly separates train vs test.
        We concatenate train/test, label them 0/1, and compute AUC(feature → is_train).
        """
        combined = pd.concat([train_df, test_df],
                             axis=0).reset_index(drop=True)
        labels = np.concatenate(
            [np.zeros(len(train_df)), np.ones(len(test_df))])
        sep_feats = []
        self.leakage_report["train_test_leakage"] = {}
        for col in numeric_cols + categorical_cols:
            try:
                x = combined[col]
                if col in categorical_cols:
                    dummies = pd.get_dummies(x, prefix=col, dummy_na=False)
                    aucs = []
                    for dummy_col in dummies.columns:
                        try:
                            auc_val = roc_auc_score(
                                labels, dummies[dummy_col].values)
                            aucs.append(auc_val)
                        except:
                            pass
                    if not aucs:
                        continue
                    max_auc = max(aucs)
                else:
                    if x.nunique() <= 1:
                        continue
                    xv = (x - x.min()) / (x.max() - x.min() + 1e-9)
                    max_auc = roc_auc_score(labels, xv.fillna(0.5))
                self.leakage_report["train_test_leakage"][col] = float(max_auc)
                if max_auc >= self.auc_threshold:
                    sep_feats.append(col)
                    if self.verbose:
                        print(
                            f"[LeakageDetection] ⚠ Feature '{col}' perfectly separates train/test (AUC={max_auc:.3f})")
            except Exception:
                continue
        self.leakage_report["train_test_leaky_features"] = sep_feats

    def dump_report(self, path: str = "leakage_report.json") -> None:
        with open(path, "w") as f:
            json.dump(self.leakage_report, f, indent=2)


class DataQualityPipeline(BaseEstimator, TransformerMixin):
    """
    Phase 2–3: Data Quality & Preprocessing Pipeline (Improved)

      0) Mixed-dtype detection & basic info
      1) Detect train/test imbalance & split
      2) Detect MCAR/MAR/MNAR via Little’s test + missingness vs target
      3) Numeric missing-value imputation (mean, median, random-sample; skip KNN/MICE if large)
      4) Univariate outlier detection & winsorization
      5) Multivariate outlier detection (Mahalanobis) & conditional row-drop
      6) Scaling (Standard, MinMax, or Robust) + before/after metrics
      7) Conditional extra transformation (Box-Cox, Yeo-Johnson, Quantile→Normal) with before/after metrics
      8) Categorical missing-value imputation (mode, constant, random-sample)
      9) Missingness flags + missingness-vs-target correlation
     10) Rare-category grouping (<1% → "__RARE__")
     11) Univariate/Bivariate feature screening (entropy, MI/χ², CPT for strong categories)
     12) Categorical encoding variants → two DataFrames (linear vs tree)
     13) PCA on numeric → optional replacement
     14) Leakage detection on TRAIN & TRAIN vs VAL
     15) Final output: (train_linear_df, train_tree_df, val_df)

    Produces:
      - analysis_report.txt
      - processed_train_linear.csv, processed_train_tree.csv
      - leakage_report.json
    """

    # -------------- Class-level constants --------------
    MIN_COMPLETE_FOR_COV = 5
    VARIANCE_RATIO_CUTOFF = 0.50
    COV_CHANGE_CUTOFF = 0.20

    UNIV_IQR_FACTOR = 1.5
    UNIV_ZSCORE_CUTOFF = 3.0
    UNIV_MODZ_CUTOFF = 3.5
    UNIV_FLAG_THRESHOLD = 2  # number of votes to call row outlier

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

    LARGE_DATASET_CUTOFF = 5000  # skip KNN/MICE if n_train > this

    IMBALANCE_WARN_RATIO = 10.0  # warn if one class is > 10× the other

    def __init__(
        self,
        target_column: Optional[str] = None,
        ordinal_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        pca_variance_threshold: float = 0.90,
        apply_pca: bool = True,
        verbose: bool = True,
        max_impute_small: int = LARGE_DATASET_CUTOFF,
        outlier_winsorize: bool = True,
    ):
        self.target = target_column
        self.ordinal_cols = set(ordinal_cols or [])
        self.test_size = test_size
        self.random_state = random_state
        self.pca_variance_threshold = pca_variance_threshold
        self.apply_pca = apply_pca
        self.verbose = verbose
        self.max_impute_small = max_impute_small
        self.outlier_winsorize = outlier_winsorize

        # Populated during fit
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None

        # Feature lists
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.ordinal_cols_present: List[str] = []
        self.nominal_cols_present: List[str] = []
        self.non_numeric_cols: List[str] = []

        # Per-feature imputers / transformers
        self.numeric_imputers: Dict[str, BaseEstimator] = {}
        self.scaler_model: Optional[TransformerMixin] = None
        self.transform_model: Dict[str,
                                   Tuple[str, Optional[TransformerMixin]]] = {}
        self.categorical_imputers: Dict[str, BaseEstimator] = {}
        self.encoding_models: Dict[str,
                                   Union[OrdinalEncoder, Dict[str, float]]] = {}
        self.pca_model: Optional[PCA] = None

        # Reporting
        self.report: Dict[str, Dict] = {
            "mixed_dtype": {},
            "imbalance": {},
            "split": {},
            "missing_mechanism": {},
            "missingness_pattern": {},
            "missing_numeric": {},
            "univariate_outliers": {},
            "multivariate_outliers": {},
            "scaler": {},
            "transform": {},
            "missingness_flags": {},
            "categorical_imputation": {},
            "rare_categories": {},
            "feature_screening": {},
            "encoding": {},
            "pca": {},
            "feature_lifetime": {},
        }
        self._report_lines: List[str] = []

        # Track when each feature is dropped/transformed
        self.feature_lifetime: Dict[str, List[str]] = {}

        # Leakage detection
        self.leakage_detector = LeakageDetection(verbose=verbose)

    # ==== 0) Mixed-Type Detection & Basic Info ====

    def _detect_mixed_types(self, df: pd.DataFrame) -> None:
        """
        Flag any column where types are mixed (int + str, float + str, etc.).
        """
        self._report_lines.append("STEP 0A: Mixed‐Type Detection")
        for col in df.columns:
            series = df[col].dropna()
            types_seen = set(type(v) for v in series.values)
            # Also catch numeric strings vs numeric:
            str_num = series.map(lambda x: str(x).replace(
                ".", "", 1).isdigit() if isinstance(x, str) else False)
            if str_num.any() and (series.map(lambda x: isinstance(x, (int, float))).any()):
                types_seen.add(str)
                types_seen.add(int)
                types_seen.add(float)
            if len(types_seen) > 1:
                self.report["mixed_dtype"][col] = [str(t) for t in types_seen]
                self._report_lines.append(
                    f"  • Column '{col}' has mixed types: {types_seen}")
            else:
                self.report["mixed_dtype"][col] = ["uniform"]
        self._report_lines.append("")

    # ==== 1) Imbalance Detection & Train/Val Split ====

    def _detect_imbalance_and_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        If a target is present and categorical, compute class ratio & warn if > threshold.
        Then split (stratified if possible).
        """
        self._report_lines.append("STEP 0B: Imbalance Check & Train/Val Split")
        if self.target and self.target in df.columns:
            y = df[self.target]
            # If binary or multiclass, compute imbalance
            if y.nunique() == 2:
                counts = y.value_counts().to_dict()
                high = max(counts.values())
                low = min(counts.values())
                ratio = high / (low + 1e-9)
                self.report["imbalance"] = {
                    "counts": counts, "ratio": float(ratio)}
                if ratio >= self.IMBALANCE_WARN_RATIO:
                    self._report_lines.append(
                        f"  • WARNING: target imbalance {high}:{low} → ratio ≈ {ratio:.1f}"
                    )
            else:
                # multiclass: record counts
                self.report["imbalance"] = {
                    "counts": y.value_counts().to_dict()}
                self._report_lines.append(
                    f"  • Multiclass target counts: {self.report['imbalance']['counts']}")
            # Now split
            tr, va = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
        else:
            tr, va = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state, stratify=None
            )
        self.report["split"] = {
            "n_total": len(df),
            "n_train": len(tr),
            "n_val": len(va),
            "test_size": self.test_size,
        }
        self._report_lines.append(
            f"  • Split into train={len(tr)}, val={len(va)}")
        return tr.reset_index(drop=True), va.reset_index(drop=True)

    # ==== 2) Missingness Mechanism + Pattern Analysis ====

    def _detect_missing_mechanism(self, df: pd.DataFrame) -> None:
        """
        Run Little’s MCAR test. Also compute missing‐vs‐target correlation
        and create missingness indicator flags.
        """
        self._report_lines.append(
            "\nSTEP 1A: Missing Mechanism (Little’s Test)")
        if self.target and self.target in df.columns:
            t = df[self.target]
            data_for_test = df.drop(columns=[self.target])
        else:
            data_for_test = df.copy()

        na_counts = data_for_test.isna().sum()
        cols_with_na = na_counts[na_counts > 0].index.tolist()
        mech_entry: Dict[str, Union[str, float]] = {}
        if len(cols_with_na) < 2:
            mech_entry["note"] = "too few NA cols → skipped MCAR test"
            self._report_lines.append(
                "  • Too few columns with missing → skipped MCAR test.")
        else:
            try:
                res = test_missingness(data_for_test[cols_with_na])
                pvalue = float(res.pvalue)
                mech = "MCAR" if pvalue > 0.05 else "MAR/MNAR"
                mech_entry = {"pvalue": pvalue, "mechanism": mech}
                self._report_lines.append(
                    f"  • Little’s MCAR p-value={pvalue:.3f} → {mech}")
            except Exception:
                mech_entry["note"] = "MCAR test failed"
                self._report_lines.append("  • MCAR test failed.")
        self.report["missing_mechanism"] = mech_entry

        # Step 1B: For each column, record missing rate and, if target exists, correlation with target
        self._report_lines.append(
            "\nSTEP 1B: Missingness Pattern & Target Correlation")
        miss_pattern: Dict[str, Dict] = {}
        for col in df.columns:
            na_rate = float(df[col].isna().mean())
            entry: Dict[str, Union[float, Dict[str, float], str]] = {
                "missing_rate": na_rate}
            if na_rate > 0:
                # create flag
                flag_col = f"{col}_is_missing"
                df[flag_col] = df[col].isna().astype(int)
                self.feature_lifetime.setdefault(
                    flag_col, []).append("created_missing_flag")
                self._report_lines.append(
                    f"  • Created missing‐flag '{flag_col}' (rate={na_rate:.3f})")
                # correlate with target if exists
                if self.target and self.target in df.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(df[self.target]):
                            # point-biserial
                            corr = stats.pointbiserialr(
                                df[flag_col], df[self.target])[0]
                        else:
                            # χ² or Cramer’s V
                            contingency = pd.crosstab(
                                df[flag_col], df[self.target])
                            chi2_stat, p, _, _ = stats.chi2_contingency(
                                contingency)
                            # Cramer's V:
                            n = contingency.sum().sum()
                            phi2 = chi2_stat / n
                            r, k = contingency.shape
                            phi2_corr = max(
                                0, phi2 - ((k - 1)*(r - 1))/(n - 1))
                            rcorr = r - ((r - 1)**2)/(n - 1)
                            kcorr = k - ((k - 1)**2)/(n - 1)
                            corr = np.sqrt(
                                phi2_corr / min((kcorr - 1), (rcorr - 1)))
                        entry["missing_vs_target_corr"] = float(corr)
                        self._report_lines.append(
                            f"    • Missingness of '{col}' vs target corr={corr:.3f}")
                    except Exception:
                        entry["missing_vs_target_corr"] = "n/a"
                else:
                    entry["missing_vs_target_corr"] = "no_target"
            else:
                entry["missing_vs_target_corr"] = 0.0
            miss_pattern[col] = entry

        # Drop any missing flags that ended up constant
        for col in list(df.columns):
            if col.endswith("_is_missing"):
                if df[col].nunique() <= 1:
                    df.drop(columns=[col], inplace=True)
                    self._report_lines.append(
                        f"  • Dropped constant missing‐flag '{col}'")
                    self.feature_lifetime[col] = ["dropped_constant_flag"]

        self.report["missingness_pattern"] = miss_pattern

    # ==== 3) Identify Feature Types & Feature Screening ====

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """
        Partition columns into numeric, ordinal, nominal, etc. Compute entropy
        and mutual info / χ² for screening vs. target.
        """
        self._report_lines.append(
            "\nSTEP 2: Feature Type Identification & Screening")
        all_cols = [c for c in df.columns if c != self.target]
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in all_cols if c not in num_cols]

        self.numeric_cols = num_cols.copy()
        # among cat_cols, split ordinal vs nominal
        self.ordinal_cols_present = [
            c for c in cat_cols if c in self.ordinal_cols]
        self.nominal_cols_present = [
            c for c in cat_cols if c not in self.ordinal_cols]
        self.categorical_cols = cat_cols.copy()
        self.non_numeric_cols = cat_cols.copy()

        self._report_lines.append(f"  • Numeric cols: {self.numeric_cols}")
        self._report_lines.append(
            f"  • Ordinal cols: {self.ordinal_cols_present}")
        self._report_lines.append(
            f"  • Nominal cols: {self.nominal_cols_present}")

        # 2A: Shannon entropy for each column
        entropy_dict: Dict[str, float] = {}
        for col in all_cols:
            try:
                freq = df[col].value_counts(
                    normalize=True, dropna=False).values
                ent = float(stats.entropy(freq))
                entropy_dict[col] = ent
                if ent < 0.1:
                    self._report_lines.append(
                        f"    - '{col}' entropy={ent:.3f} ≈ 0 (near-constant)")
            except Exception:
                entropy_dict[col] = np.nan
        self.report["feature_screening"] = {"entropy": entropy_dict}

        # 2B: Mutual information vs. target (if supervised)
        if self.target and self.target in df.columns:
            mi_dict: Dict[str, float] = {}
            if pd.api.types.is_numeric_dtype(df[self.target]):
                # regression MI
                y = df[self.target]
                for col in self.numeric_cols:
                    try:
                        mi = float(mutual_info_regression(df[[col]].fillna(
                            df[col].mean()), y.fillna(y.mean()), random_state=self.random_state)[0])
                        mi_dict[col] = mi
                    except:
                        mi_dict[col] = 0.0
                for col in self.nominal_cols_present:
                    try:
                        # encode as integer codes
                        codes = df[col].astype(
                            "category").cat.codes.fillna(-1).values.reshape(-1, 1)
                        mi = float(mutual_info_regression(codes, y.fillna(
                            y.mean()), random_state=self.random_state)[0])
                        mi_dict[col] = mi
                    except:
                        mi_dict[col] = 0.0
            else:
                # classification MI (discrete target)
                y = df[self.target].astype("category").cat.codes
                for col in self.numeric_cols:
                    try:
                        mi = float(mutual_info_classif(df[[col]].fillna(
                            df[col].mean()), y, random_state=self.random_state)[0])
                        mi_dict[col] = mi
                    except:
                        mi_dict[col] = 0.0
                for col in self.nominal_cols_present:
                    try:
                        codes = df[col].astype(
                            "category").cat.codes.values.reshape(-1, 1)
                        mi = float(mutual_info_classif(
                            codes, y, random_state=self.random_state)[0])
                        mi_dict[col] = mi
                    except:
                        mi_dict[col] = 0.0
            self.report["feature_screening"]["mutual_info"] = mi_dict
            # Flag low‐MI features (e.g. bottom 10%)
            threshold = np.percentile(
                [v for v in mi_dict.values() if not np.isnan(v)], 10)
            low_mi = [col for col, v in mi_dict.items() if v <= threshold]
            self.report["feature_screening"]["low_mi_features"] = low_mi
            if low_mi:
                self._report_lines.append(
                    f"  • Low‐MI features (<=10th percentile): {low_mi}")
        else:
            self.report["feature_screening"]["mutual_info"] = "no_target"

        # 2C: Bivariate (numeric–numeric) correlation suggestions
        if len(self.numeric_cols) >= 2:
            corr_matrix = df[self.numeric_cols].corr().abs().fillna(0)
            # Extract top 20 pairs with highest |corr| (excluding diagonal)
            pairs = []
            cols = self.numeric_cols
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
            pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
            top_nn = pairs_sorted[:20]
            suggestions = [
                f"{a} × {b} (|corr|={c:.2f})" for a, b, c in top_nn if c >= 0.3]
            self.report["feature_screening"]["top_numeric_pairs"] = top_nn
            if suggestions:
                self._report_lines.append(
                    "  • Suggested numeric interactions (|corr| ≥ 0.3):")
                for s in suggestions:
                    self._report_lines.append(f"      - {s}")
        else:
            self.report["feature_screening"]["top_numeric_pairs"] = []

        # 2D: Bivariate (numeric–categorical) MI suggestions
        if self.target and self.target in df.columns and self.numeric_cols and self.nominal_cols_present:
            bic_mi = []
            y = df[self.target].astype("category").cat.codes if not pd.api.types.is_numeric_dtype(
                df[self.target]) else df[self.target]
            for num in self.numeric_cols:
                for cat in self.nominal_cols_present:
                    try:
                        # discretize numeric into 10 bins
                        bins = pd.qcut(df[num].fillna(
                            df[num].median()), q=10, duplicates='drop', labels=False)
                        contingency = pd.crosstab(
                            bins, df[cat].astype("category").cat.codes)
                        chi2_stat, p, _, _ = stats.chi2_contingency(
                            contingency)
                        bic_mi.append((num, cat, chi2_stat))
                    except:
                        bic_mi.append((num, cat, 0.0))
            bic_sorted = sorted(bic_mi, key=lambda x: x[2], reverse=True)[:20]
            self.report["feature_screening"]["top_num_cat_pairs"] = bic_sorted
            if bic_sorted and bic_sorted[0][2] >= 50:  # arbitrary large chi2
                self._report_lines.append(
                    "  • Suggested num×cat interactions (χ² heuristic):")
                for num, cat, chi2_stat in bic_sorted[:5]:
                    self._report_lines.append(
                        f"      - {num} vs {cat} (χ²={chi2_stat:.1f})")
        else:
            self.report["feature_screening"]["top_num_cat_pairs"] = []

    # ==== 4) Numeric Missing‐Value Imputation ====

    def _random_sample_impute_num(self, orig: pd.Series) -> pd.Series:
        nonnull = orig.dropna().values
        out = orig.copy()
        mask = out.isna()
        if len(nonnull) == 0:
            return out.fillna(0.0)
        out.loc[mask] = np.random.choice(
            nonnull, size=mask.sum(), replace=True
        )
        return out

    def _evaluate_impute_num(
        self, col: str, orig: pd.Series, imputed: pd.Series, cov_before: Optional[np.ndarray]
    ) -> Tuple[float, float, float]:
        orig_nonnull = orig.dropna().values
        imp_nonnull = imputed.dropna().values

        # KS p-value
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
        Impute numeric columns on TRAIN with:
          • Mean / Median
          • (Optionally KNN / MICE if n_train <= max_impute_small)
          • Random-sample fallback
        Choose best by KS p-value > 0, var_ratio ≥ 0.5, cov_change ≤ 0.2.
        """
        self._report_lines.append("\nSTEP 3: Numeric Missing-Value Imputation")
        df = self.train_df
        df_num = df[self.numeric_cols].copy()
        complete_df = df_num.dropna()
        if complete_df.shape[0] < self.MIN_COMPLETE_FOR_COV:
            cov_before = None
            self._report_lines.append(
                "  • Too few complete rows → skipping covariance QC")
        else:
            cov_before = EmpiricalCovariance().fit(complete_df.values).covariance_

        n_train = df.shape[0]
        use_knn_mice = n_train <= self.max_impute_small

        for col in self.numeric_cols:
            orig = df_num[col]
            n_missing = int(orig.isna().sum())
            if n_missing == 0:
                self.report["missing_numeric"][col] = {
                    "chosen": "none", "note": "no missing"}
                self._report_lines.append(f"  • {col:20s} → no missing")
                continue

            candidates: Dict[str, pd.Series] = {}
            imputers: Dict[str, Optional[BaseEstimator]] = {}
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

            # 3) KNN (only if small)
            if use_knn_mice:
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
                    imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                                           random_state=self.random_state, max_iter=10)
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
            else:
                self._report_lines.append(
                    f"  • Skipping KNN/MICE on '{col}' (n_train={n_train} > {self.max_impute_small})")

            # 5) Random-sample
            try:
                arr = self._random_sample_impute_num(orig)
                ksp, vr, cc = self._evaluate_impute_num(
                    col, orig, arr, cov_before)
                candidates["random_sample"] = arr
                imputers["random_sample"] = None
                metrics["random_sample"] = (ksp, vr, cc)
            except:
                pass

            # Choose best
            best_method = None
            best_score = (-1.0, -1.0, np.inf)
            for m, (ksp, vr, cc) in metrics.items():
                if (ksp > best_score[0]
                    and vr >= self.VARIANCE_RATIO_CUTOFF
                        and (np.isnan(cc) or cc <= self.COV_CHANGE_CUTOFF)):
                    best_method = m
                    best_score = (ksp, vr, cc)

            if best_method is None:
                # fallback to mean
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
                self.feature_lifetime.setdefault(
                    col, []).append("imputed_fallback_mean")
            else:
                self.report["missing_numeric"][col] = {
                    "chosen": best_method,
                    "metrics": best_score,
                    "note": ""
                }
                self._report_lines.append(
                    f"  • {col:20s} → {best_method}, metrics={best_score}")
                self.train_df[col] = candidates[best_method]
                self.numeric_imputers[col] = imputers[best_method]
                self.feature_lifetime.setdefault(
                    col, []).append(f"imputed_{best_method}")

    # ==== 5) Univariate Outlier Detection & Conditional Winsorization / Row‐Drop ====

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

    def _detect_and_handle_univariate_outliers(self) -> None:
        """
        Vote-based outlier detection for each numeric column.
        If outlier_winsorize=True, winsorize that cell to nearest bound; else, accumulate vote.
        After voting, any row with votes >= threshold gets dropped entirely.
        """
        self._report_lines.append(
            "\nSTEP 4A: Univariate Outlier Detection & Handling")
        df = self.train_df
        univ_votes: Dict[int, int] = {}

        for col in self.numeric_cols:
            self.report["univariate_outliers"][col] = {}
            s = df[col]
            # store pre‐skew/kurt/quantiles
            pre_skew, pre_kurt = float(stats.skew(s.dropna())), float(
                stats.kurtosis(s.dropna()))
            q1, q3 = np.percentile(s.dropna(), [25, 75]) if s.dropna(
            ).size >= 2 else (np.nan, np.nan)
            lb = q1 - self.UNIV_IQR_FACTOR * \
                (q3 - q1) if not np.isnan(q1) and not np.isnan(q3) else np.nan
            ub = q3 + self.UNIV_IQR_FACTOR * \
                (q3 - q1) if not np.isnan(q1) and not np.isnan(q3) else np.nan

            outlier_indices = set()
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
                        if self.outlier_winsorize:
                            # winsorize that value to nearest bound
                            original_val = df.at[i, col]
                            if pd.isna(original_val):
                                continue
                            new_val = lb if original_val < lb else (
                                ub if original_val > ub else original_val)
                            df.at[i, col] = new_val
                            self.feature_lifetime.setdefault(col, []).append(
                                f"winsorized_{method_name}_idx{i}")
                        else:
                            univ_votes[i] = univ_votes.get(i, 0) + 1
                            self.feature_lifetime.setdefault(col, []).append(
                                f"flagged_{method_name}_idx{i}")
                except:
                    self.report["univariate_outliers"][col][method_name] = None
                    self._report_lines.append(
                        f"  • {col:20s} via {method_name:>6s} → error")

            # after winsorization, record post‐skew/kurt
            post_skew, post_kurt = float(stats.skew(df[col].dropna())), float(
                stats.kurtosis(df[col].dropna()))
            self.report["univariate_outliers"][col]["pre_skew"] = pre_skew
            self.report["univariate_outliers"][col]["post_skew"] = post_skew
            self.report["univariate_outliers"][col]["pre_kurt"] = pre_kurt
            self.report["univariate_outliers"][col]["post_kurt"] = post_kurt

        # Now handle row‐drop if winsorize=False or too many flags
        if not self.outlier_winsorize:
            self._report_lines.append("\nSTEP 4B: Dropping Real Outlier Rows")
            idxs_to_drop = [idx for idx, v in univ_votes.items(
            ) if v >= self.UNIV_FLAG_THRESHOLD]
            self.report["multivariate_outliers"]["univariate_drops"] = idxs_to_drop
            self._report_lines.append(
                f"  • Dropping {len(idxs_to_drop)} rows with ≥{self.UNIV_FLAG_THRESHOLD} votes")
            if idxs_to_drop:
                df.drop(index=idxs_to_drop, inplace=True)
                df.reset_index(drop=True, inplace=True)
                for idx in idxs_to_drop:
                    for col in self.numeric_cols:
                        self.feature_lifetime.setdefault(
                            col, []).append(f"dropped_row_idx{idx}")

    # ==== 6) Multivariate Outlier Detection (Mahalanobis) ====

    def _detect_multivariate_outliers(self) -> None:
        """
        Compute Mahalanobis distance on complete numeric instances.
        Flag rows where D^2 > χ²_{p,0.975}; either winsorize each numeric coordinate
        or drop the row depending on outlier_winsorize setting.
        """
        self._report_lines.append(
            "\nSTEP 5: Multivariate Outlier Detection (Mahalanobis)")
        df = self.train_df
        p = len(self.numeric_cols)
        numeric_only = df[self.numeric_cols].dropna()
        if numeric_only.shape[0] >= max(self.UNIV_FLAG_THRESHOLD * p, self.MIN_COMPLETE_FOR_COV):
            cov = EmpiricalCovariance().fit(numeric_only.values)
            md = cov.mahalanobis(numeric_only.values)
            thresh = chi2.ppf(self.MULTI_CI, df=p)
            mask = md > thresh
            idxs = numeric_only.index[mask]
            self.report["multivariate_outliers"]["mahalanobis"] = list(idxs)
            self._report_lines.append(f"  • Mahalanobis → {len(idxs)} flagged")
            if self.outlier_winsorize:
                # winsorize each numeric: any coordinate outside ±sqrt(thresh) * std → cap at boundary
                stds = numeric_only.std(axis=0).values
                center = numeric_only.mean(axis=0).values
                boundary = np.sqrt(thresh)
                for i in idxs:
                    row = df.loc[i, self.numeric_cols].values
                    z = (row - center) / stds
                    new_row = np.where(
                        np.abs(z) > boundary, center + np.sign(z) * boundary * stds, row)
                    for j, col in enumerate(self.numeric_cols):
                        if row[j] != new_row[j]:
                            df.at[i, col] = new_row[j]
                            self.feature_lifetime.setdefault(col, []).append(
                                f"winsorized_mahalanobis_idx{i}")
            else:
                self._report_lines.append(
                    f"  • Dropping {len(idxs)} rows due to Mahalanobis")
                df.drop(index=list(idxs), inplace=True)
                df.reset_index(drop=True, inplace=True)
                for col in self.numeric_cols:
                    for i in idxs:
                        self.feature_lifetime.setdefault(
                            col, []).append(f"dropped_row_idx{i}")
        else:
            self.report["multivariate_outliers"]["mahalanobis"] = []
            self._report_lines.append(
                "  • Not enough complete rows → skipped Mahalanobis")

    # ==== 7) Scaling Numeric with Before/After Metrics ====

    def _choose_scaler(self) -> Tuple[str, Dict[str, Dict[str, float]]]:
        """
        Choose Standard vs Robust vs MinMax based on skew/kurtosis of each numeric
        on TRAIN (after imputation/outlier handling).
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

        if any(abs(sk) > self.SKEW_THRESH_FOR_ROBUST or abs(ku) > self.KURT_THRESH_FOR_ROBUST
               for sk, ku in zip(skews.values(), kurts.values())):
            return "RobustScaler", {"skew": skews, "kurtosis": kurts}
        if all(abs(sk) < self.SKEW_THRESH_FOR_STANDARD for sk in skews.values()):
            return "StandardScaler", {"skew": skews, "kurtosis": kurts}
        return "MinMaxScaler", {"skew": skews, "kurtosis": kurts}

    def _apply_scaling(self) -> None:
        """
        Fit & apply the chosen scaler on TRAIN numeric. Record
        before/after skew/kurtosis/Shapiro p-values.
        """
        self._report_lines.append("\nSTEP 6: Scaling (Numeric)")
        if not self.numeric_cols:
            self.report["scaler"] = {
                "chosen": "none", "note": "no numeric cols"}
            self._report_lines.append("  • No numeric columns to scale.")
            return

        # Compute pre‐scaling metrics
        pre_metrics: Dict[str, Dict[str, float]] = {}
        for col in self.numeric_cols:
            arr = self.train_df[col].dropna().values
            if len(arr) >= 3:
                skew_val = float(stats.skew(arr))
                kurt_val = float(stats.kurtosis(arr))
                try:
                    sh_p = float(stats.shapiro(arr)[1])
                except:
                    sh_p = 0.0
                pre_metrics[col] = {"skew": skew_val,
                                    "kurtosis": kurt_val, "shapiro_p": sh_p}
            else:
                pre_metrics[col] = {"skew": 0.0,
                                    "kurtosis": 0.0, "shapiro_p": 1.0}

        scaler_name, metrics = self._choose_scaler()
        self.report["scaler"] = {"chosen": scaler_name,
                                 "stats": metrics, "pre_metrics": pre_metrics}
        self._report_lines.append(f"  • Suggested scaler: {scaler_name}")

        if scaler_name == "StandardScaler":
            model = StandardScaler()
        elif scaler_name == "MinMaxScaler":
            model = MinMaxScaler()
        else:
            model = RobustScaler()

        scaled_arr = model.fit_transform(self.train_df[self.numeric_cols])
        self.train_df[self.numeric_cols] = scaled_arr
        self.scaler_model = clone(model)
        self.feature_lifetime.setdefault("scaler", []).append(scaler_name)
        self._report_lines.append(f"  • Applied {scaler_name} on TRAIN")

        # Compute post‐scaling metrics
        post_metrics: Dict[str, Dict[str, float]] = {}
        for idx, col in enumerate(self.numeric_cols):
            arr = self.train_df[col].dropna().values
            if len(arr) >= 3:
                skew_val = float(stats.skew(arr))
                kurt_val = float(stats.kurtosis(arr))
                try:
                    sh_p = float(stats.shapiro(arr)[1])
                except:
                    sh_p = 0.0
                post_metrics[col] = {"skew": skew_val,
                                     "kurtosis": kurt_val, "shapiro_p": sh_p}
            else:
                post_metrics[col] = {"skew": 0.0,
                                     "kurtosis": 0.0, "shapiro_p": 1.0}
        self.report["scaler"]["post_metrics"] = post_metrics

    # ==== 8) Conditional Extra Transformation (Box‐Cox / Yeo‐Johnson / Quantile) ====

    def _evaluate_transform(
        self, arr_scaled: np.ndarray
    ) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """
        Evaluate transforms (none, boxcox, yeo, quantile) on a scaled array.
        Score by (Shapiro p-value, -|skew|) → want higher p and lower skew.
        """
        scores: Dict[str, Tuple[float, float]] = {}
        if len(arr_scaled) < 5 or np.unique(arr_scaled).size < 5:
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

                sample = arr if arr.size <= 5000 else np.random.choice(
                    arr, 5000, replace=False)
                if sample.size >= 3:
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

    def _apply_extra_transform(self) -> None:
        """
        For each numeric, if post-scaling Shapiro p < 0.05 or |skew| > 0.75,
        evaluate extra transforms and pick the best. Record before/after metrics.
        """
        self._report_lines.append(
            "\nSTEP 7: Extra Numeric Transform (Conditional)")
        for col in self.numeric_cols:
            arr = self.train_df[col].dropna().values
            if len(arr) < 5:
                self.report["transform"][col] = {
                    "chosen": "none",
                    "scores": {"none": (1.0, 0.0)},
                    "note": "too few samples"
                }
                self._report_lines.append(
                    f"  • {col:20s} → too few for transform")
                continue

            # Check post‐scaling normality
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
                    "chosen": "none",
                    "scores": {"none": (pval_scaled, skew_scaled)},
                    "note": "scaling sufficed"
                }
                self._report_lines.append(
                    f"  • {col:20s} → no transform (scaling OK)")
                continue

            best, scores = self._evaluate_transform(arr)
            self.report["transform"][col] = {
                "chosen": best,
                "scores": scores,
                "pre_metrics": {"skew": float(stats.skew(arr)), "shapiro_p": float(stats.shapiro(arr)[1]) if len(arr) >= 3 else 0.0}
            }
            self._report_lines.append(f"  • {col:20s} → evaluating transforms")
            self._report_lines.append(f"    • scores: {scores}")
            if best == "boxcox":
                arr_trans, _ = stats.boxcox(arr)
                self.train_df[col].iloc[:] = pd.Series(
                    arr_trans, index=self.train_df[self.numeric_cols].dropna().index)
                self.transform_model[col] = ("boxcox", None)
                self.feature_lifetime.setdefault(
                    col, []).append("transformed_boxcox")
                self._report_lines.append(f"    • Applied Box-Cox to '{col}'.")
            elif best == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                transformed = pt.fit_transform(arr.reshape(-1, 1)).flatten()
                idxs = self.train_df[self.train_df[col].notna()].index
                for i, idx in enumerate(idxs):
                    self.train_df.at[idx, col] = transformed[i]
                self.transform_model[col] = ("yeo", clone(pt))
                self.feature_lifetime.setdefault(
                    col, []).append("transformed_yeo")
                self._report_lines.append(
                    f"    • Applied Yeo-Johnson to '{col}'.")
            elif best == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal", random_state=self.random_state)
                transformed = qt.fit_transform(arr.reshape(-1, 1)).flatten()
                idxs = self.train_df[self.train_df[col].notna()].index
                for i, idx in enumerate(idxs):
                    self.train_df.at[idx, col] = transformed[i]
                self.transform_model[col] = ("quantile", clone(qt))
                self.feature_lifetime.setdefault(
                    col, []).append("transformed_quantile")
                self._report_lines.append(
                    f"    • Applied Quantile→Normal to '{col}'.")
            else:
                self._report_lines.append(
                    f"    • No extra transform for '{col}'.")

            # Post‐transform metrics
            arr2 = self.train_df[col].dropna().values
            post_skew = float(stats.skew(arr2))
            post_p = float(stats.shapiro(arr2)[1]) if len(arr2) >= 3 else 0.0
            self.report["transform"][col]["post_metrics"] = {
                "skew": post_skew, "shapiro_p": post_p}

    # ==== 9) Categorical Missing‐Value Imputation ====

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
        Impute categorical columns on TRAIN by minimizing TVD.
        """
        self._report_lines.append(
            "\nSTEP 8: Categorical Missing-Value Imputation")
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
                self.feature_lifetime.setdefault(
                    col, []).append("imputed_mode")
                self._report_lines.append(
                    f"    • Applied mode imputation to '{col}'.")
            elif best == "constant_other":
                df[col] = orig.fillna("__MISSING__").astype(object)
                self.categorical_imputers[col] = "__MISSING__"
                self.feature_lifetime.setdefault(
                    col, []).append("imputed_constant")
                self._report_lines.append(
                    f"    • Filled nulls with '__MISSING__' in '{col}'.")
            elif best == "random_sample":
                df[col] = self._random_sample_impute_cat(orig)
                self.categorical_imputers[col] = None
                self.feature_lifetime.setdefault(
                    col, []).append("imputed_random")
                self._report_lines.append(
                    f"    • Random-sample imputed '{col}'.")
            else:
                df[col] = orig.fillna("__MISSING__").astype(object)
                self.categorical_imputers[col] = "__MISSING__"
                self.feature_lifetime.setdefault(
                    col, []).append("imputed_fallback_constant")
                self._report_lines.append(
                    f"    • Fallback: filled '__MISSING__' for '{col}'.")

    # ==== 10) Rare‐Category Grouping ====

    def _group_rare_categories(self) -> None:
        """
        Group categories with freq < RARE_FREQ_CUTOFF into "__RARE__".
        """
        self._report_lines.append("\nSTEP 9: Rare-Category Grouping")
        df = self.train_df
        n_rows = len(df)
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
            self.feature_lifetime.setdefault(col, []).append("grouped_rare")

    # ==== 11) Categorical Encoding Variants (Linear vs Tree) ====

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

    def _encode_categorical_variants(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build two separate DataFrames:
          • Linear: one-hot (if low cardinality) + freq
          • Tree: one-hot (small), ordinal (medium), freq (larger)
        Return (train_linear_df, train_tree_df), and save CSVs.
        """
        self._report_lines.append("\nSTEP 10: Categorical Encoding Variants")
        base = self.train_df.copy()
        n_rows = len(base)

        # 10A) LINEAR: if frac_unique ≤ ONEHOT_MAX_UNIQ → one-hot; else freq
        linear_onehot, linear_freq, linear_sugg = [], [], []
        for col in self.categorical_cols:
            uniq = base[col].nunique()
            frac = uniq / n_rows if n_rows > 0 else 0.0
            if col in self.ordinal_cols_present:
                # force ordinal rather than one-hot
                linear_sugg.append(
                    f"[LINEAR] '{col}' is ordinal → consider ordinal enc")
                linear_freq.append(col)
            elif frac <= self.ONEHOT_MAX_UNIQ:
                linear_onehot.append(col)
            else:
                if frac <= self.KNOWNALLY_MAX_UNIQ:
                    linear_freq.append(col)
                else:
                    linear_sugg.append(
                        f"[LINEAR] '{col}' frac={frac:.2f} > {self.ULTRAHIGH_UNIQ}: suggest target encoding"
                    )

        oh = self._onehot_encode(
            base, linear_onehot) if linear_onehot else pd.DataFrame(index=base.index)
        freq_df = self._frequency_encode(
            base, linear_freq) if linear_freq else pd.DataFrame(index=base.index)
        df_lin = pd.concat(
            [base.drop(columns=self.categorical_cols), oh, freq_df], axis=1)
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

        # 10B) TREE: if frac ≤ ONEHOT_MAX_UNIQ → one-hot,
        # elif ≤ ORDINAL_MAX_UNIQ → ordinal,
        # elif ≤ KNOWNALLY_MAX_UNIQ → freq,
        # else suggest target
        tree_onehot, tree_ordinal, tree_freq, tree_sugg = [], [], [], []
        df_tree = base.copy()
        for col in self.categorical_cols:
            uniq = base[col].nunique()
            frac = uniq / n_rows if n_rows > 0 else 0.0
            if col in self.ordinal_cols_present:
                tree_ordinal.append(col)
            elif frac <= self.ONEHOT_MAX_UNIQ:
                tree_onehot.append(col)
            elif frac <= self.ORDINAL_MAX_UNIQ:
                tree_ordinal.append(col)
            elif frac <= self.KNOWNALLY_MAX_UNIQ:
                tree_freq.append(col)
            else:
                tree_sugg.append(
                    f"[TREE] '{col}' frac={frac:.2f} > {self.ULTRAHIGH_UNIQ}: suggest target encoding"
                )

        # Apply one-hot
        if tree_onehot:
            oh2 = self._onehot_encode(base, tree_onehot)
            df_tree.drop(columns=tree_onehot, inplace=True)
            df_tree = pd.concat([df_tree, oh2], axis=1)
        # Apply ordinal
        if tree_ordinal:
            ord_df = self._ordinal_encode(df_tree, tree_ordinal)
            df_tree.drop(columns=tree_ordinal, inplace=True)
            df_tree = pd.concat([df_tree, ord_df], axis=1)
        # Apply freq
        if tree_freq:
            freq_df2 = self._frequency_encode(df_tree, tree_freq)
            df_tree.drop(columns=tree_freq, inplace=True)
            df_tree = pd.concat([df_tree, freq_df2], axis=1)

        df_tree.to_csv("processed_train_tree.csv", index=False)
        self.report["encoding"]["tree"] = {
            "onehot": tree_onehot,
            "ordinal": tree_ordinal,
            "frequency": tree_freq,
            "suggestions": tree_sugg,
        }
        self._report_lines.append(
            f"  • [TREE] onehot={tree_onehot}, ordinal={tree_ordinal}, freq={tree_freq}")
        for s in tree_sugg:
            self._report_lines.append(f"    • {s}")

        return df_lin, df_tree

    # ==== 12) PCA on Numeric (Optional) ====

    def _apply_pca(self) -> None:
        """
        Run PCA on TRAIN numeric columns if requested. Replace them with PCs.
        """
        if not self.apply_pca:
            self.report["pca"] = {"note": "skipped (apply_pca=False)"}
            self._report_lines.append(
                "\nSTEP 11: PCA skipped (apply_pca=False)")
            return

        self._report_lines.append("\nSTEP 11: PCA on Numeric Columns")
        X = self.train_df[self.numeric_cols].copy()
        if X.isna().any(axis=None):
            self._report_lines.append("  • PCA aborted: NaNs present")
            self.report["pca"] = {"note": "skipped (NaNs)"}
            return

        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < 5 * n_feats:
            self._report_lines.append(
                f"  • PCA aborted: too few (n={n_samples}, p={n_feats})")
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
            f"  • PCA chooses {n_comp}/{n_feats} comps (cumvar={cumvar[n_comp - 1]:.3f})")

        pca_model = PCA(n_components=n_comp)
        X_reduced = pca_model.fit_transform(X_std)
        cols = [f"PC{i+1}" for i in range(n_comp)]
        df_reduced = pd.DataFrame(
            X_reduced, columns=cols, index=self.train_df.index)

        self.train_df = pd.concat(
            [self.train_df.drop(columns=self.numeric_cols), df_reduced], axis=1)
        self.pca_model = pca_model
        self.feature_lifetime.setdefault("pca", []).append(
            f"replaced_{n_comp}_components")

    # ==== 13) Leakage Detection on TRAIN and TRAIN vs VAL ====

    def _run_leakage_checks(self) -> None:
        """
        Check for target leakage (TRAIN only) and train/val leakage.
        Save a JSON report.
        """
        self._report_lines.append("\nSTEP 12: Leakage Detection")
        if self.target and self.target in self.train_df.columns:
            # Target leakage: pass DataFrame (with features) and target series
            self.leakage_detector.check_target_leakage(
                self.train_df, self.train_df[self.target] if self.target in self.train_df.columns else self.val_df[self.target],
                self.categorical_cols, self.numeric_cols
            )
        else:
            self._report_lines.append(
                "  • No target → skipping target‐leakage check")

        # Train vs Val
        combined_train = self.train_df.drop(columns=[
                                            self.target]) if self.target and self.target in self.train_df.columns else self.train_df
        combined_val = self.val_df.drop(columns=[
                                        self.target]) if self.target and self.target in self.val_df.columns else self.val_df
        self.leakage_detector.check_train_test_separation(
            combined_train, combined_val, self.numeric_cols, self.categorical_cols
        )
        self.leakage_detector.dump_report("leakage_report.json")
        self._report_lines.append(
            "  • Leakage report saved to 'leakage_report.json'")

    # ==== Public API: fit_transform ====

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        1) Mixed‐dtype detection
        2) Imbalance check & train/val split
        3) Missing mechanism & missingness pattern
        4) Identify feature types & feature screening
        5) Impute numeric missing
        6) Univariate outlier detection & handling
        7) Multivariate outlier detection & handling
        8) Scale numeric + before/after metrics
        9) Extra transforms (Box‐Cox / Yeo / Quantile)
       10) Impute categorical missing
       11) Rare‐category grouping
       12) Feature screening (already partly done)
       13) Categorical encoding variants → two DataFrames
       14) PCA (optional)
       15) Leakage detection
       16) Return (train_linear_df, train_tree_df, val_df)
        """
        if isinstance(X, np.ndarray):
            raise ValueError("Input must be a pandas DataFrame")
        df = X.copy()
        if self.target and self.target in df.columns:
            # Keep target for splitting
            train_raw, val_raw = self._detect_imbalance_and_split(df)
            self.train_df = train_raw.reset_index(drop=True)
            self.val_df = val_raw.reset_index(drop=True)
        else:
            train_raw, val_raw = self._detect_imbalance_and_split(df)
            self.train_df = train_raw.reset_index(drop=True)
            self.val_df = val_raw.reset_index(drop=True)

        # 0A: Mixed‐type detection
        self._detect_mixed_types(self.train_df)

        # 1A,1B: Missing mechanism & pattern (on TRAIN)
        self._detect_missing_mechanism(self.train_df)

        # 2: Identify feature types & screening
        self._identify_feature_types(self.train_df)

        # 3: Numeric missing imputation
        if self.numeric_cols:
            self._impute_missing_numeric()

        # 4: Univariate outlier handling
        if self.numeric_cols:
            self._detect_and_handle_univariate_outliers()

        # 5: Multivariate outlier handling
        if self.numeric_cols:
            self._detect_multivariate_outliers()

        # 6: Scaling numeric
        if self.numeric_cols:
            self._apply_scaling()

        # 7: Extra transforms (Box‐Cox / Yeo / Quantile)
        if self.numeric_cols:
            self._apply_extra_transform()

        # 8: Categorical missing imputation
        if self.categorical_cols:
            self._impute_missing_categorical()

        # 9: Rare‐category grouping
        if self.categorical_cols:
            self._group_rare_categories()

        # 10: Categorical encoding variants → produce two DataFrames
        train_lin_df, train_tree_df = self._encode_categorical_variants()

        # 11: PCA
        if self.numeric_cols:
            self._apply_pca()

        # 12: Leakage detection
        self._run_leakage_checks()

        # 13: Write analysis report
        Path("analysis_report.txt").write_text("\n".join(self._report_lines))

        # 14: Validate VAL distributions vs TRAIN (scaler/transform range checks)
        self._validate_val_distribution(
            train_lin_df, train_tree_df, self.val_df)

        return train_lin_df.copy(), train_tree_df.copy(), self.val_df.copy()

    def _validate_val_distribution(
        self, train_lin: pd.DataFrame, train_tree: pd.DataFrame, val_df: pd.DataFrame
    ) -> None:
        """
        For each numeric column in VAL, check if:
          • After TRAIN‐scaler: any VAL features lie outside [0,1] for MinMax
          • After Box-Cox: any negative values (if Box‐Cox applied)
          • After Yeo: any NaNs if input was zero?
        Log warnings in report.
        """
        self._report_lines.append("\nSTEP 13: VAL Distribution Validation")
        # Only if scaler exists
        if self.scaler_model:
            for col in self.numeric_cols:
                try:
                    val_col = val_df[col].fillna(
                        self.train_df[col].mean()).values.reshape(-1, 1)
                    scaled = self.scaler_model.transform(val_col).flatten()
                    if np.any(scaled < -1e-6) or np.any(scaled > 1 + 1e-6):
                        self._report_lines.append(
                            f"  • WARNING: VAL '{col}' values out-of-range after scaling")
                        self.report.setdefault("val_distribution_issues", []).append(
                            {"feature": col, "issue": "scaled_out_of_range"}
                        )
                except Exception:
                    pass
        # Check extra transforms
        for col, (method, obj) in self.transform_model.items():
            if method == "boxcox":
                # Box-Cox expects positive; any VAL <= 0?
                if (val_df[col] <= 0).any():
                    self._report_lines.append(
                        f"  • WARNING: VAL '{col}' has ≤0 values for Box-Cox")
                    self.report.setdefault("val_distribution_issues", []).append(
                        {"feature": col, "issue": "boxcox_nonpositive_vals"}
                    )
            elif method == "yeo":
                try:
                    val_col = val_df[col].fillna(
                        self.train_df[col].median()).values.reshape(-1, 1)
                    transformed = obj.transform(val_col).flatten()
                    if np.any(np.isnan(transformed)):
                        self._report_lines.append(
                            f"  • WARNING: VAL '{col}' Yeo-Johnson produced NaNs")
                        self.report.setdefault("val_distribution_issues", []).append(
                            {"feature": col, "issue": "yeo_nan_vals"}
                        )
                except Exception:
                    pass
            elif method == "quantile":
                try:
                    val_col = val_df[col].fillna(
                        self.train_df[col].median()).values.reshape(-1, 1)
                    transformed = obj.transform(val_col).flatten()
                    if np.any(np.isnan(transformed)):
                        self._report_lines.append(
                            f"  • WARNING: VAL '{col}' Quantile produced NaNs")
                        self.report.setdefault("val_distribution_issues", []).append(
                            {"feature": col, "issue": "quantile_nan_vals"}
                        )
                except Exception:
                    pass

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
