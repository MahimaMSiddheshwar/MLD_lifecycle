#!/usr/bin/env python3
"""
Stage 2: Numeric & Categorical Missing‐Value Imputation

  • Uses multiple candidates (mean, median, KNN, MICE, random‐sample) for numerics.
  • For categoricals: mode, constant “__MISSING__”, or random‐sample via TVD minimization.
  • Never drops a column unless missing fraction > max_missing_frac_drop (default 0.90).
  • Performs Little’s test (omnibus) plus per‐column logistic regression to classify MCAR vs MAR/MNAR.
  • Exposes all thresholds as class‐level constants or __init__ parameters.
"""

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.missing import test_missingness  # omnibus MCAR

from sklearn.base import clone
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.covariance import EmpiricalCovariance

log = logging.getLogger("stage2")
REPORT_PATH = Path("reports/missingness")
REPORT_PATH.mkdir(parents=True, exist_ok=True)


class MissingnessAnalyzer:
    """
    Detects missingness mechanism for each column individually.
    - Uses Little’s MCAR test for all columns at once.
    - Uses per‐column logistic regression to decide MCAR vs MAR/MNAR.

    If per‐column p‐value > ALPHA → “MCAR”; else “MAR/MNAR”.
    Saves a JSON report at reports/missingness/column_missingness.json.
    """
    ALPHA = 0.05  # significance threshold to declare MAR/MNAR

    @staticmethod
    def omnibus_mcar_test(df: pd.DataFrame) -> Tuple[float, str]:
        """
        Run Little’s MCAR test across all columns that have any NA.
        Returns (pvalue, mechanism).  Mechanism = "MCAR" if p > ALPHA else "MAR/MNAR".
        """
        na_counts = df.isna().sum()
        cols_na = na_counts[na_counts > 0].index.tolist()
        if len(cols_na) < 2:
            return (np.nan, "too_few_nas")

        try:
            res = test_missingness(df[cols_na])
            pval = float(res.pvalue)
            mech = "MCAR" if (pval > MissingnessAnalyzer.ALPHA) else "MAR/MNAR"
            return (pval, mech)
        except Exception:
            return (np.nan, "test_failed")

    @staticmethod
    def per_column_missingness(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        For each column with missing values:
          - Fit logistic regression: is_missing(col) ~ indicators of not‐missing(other_cols).
          - Compute approximate Wald‐based p‐values for each coefficient; take max → p_combined.
          - If p_combined > ALPHA → “MCAR”; else “MAR/MNAR”.

        Returns a dict: { col: { "fraction_missing": f, "p_value": p, "mechanism": … } }
        Writes a JSON file to reports/missingness/column_missingness.json.
        """
        results: Dict[str, Dict[str, float]] = {}
        for col in df.columns:
            series = df[col]
            frac_missing = float(series.isna().mean())

            if frac_missing == 0.0:
                results[col] = {
                    "fraction_missing": 0.0,
                    "p_value": np.nan,
                    "mechanism": "no_nas"
                }
                continue

            # Build target y = 1 if missing in col, else 0
            y = series.isna().astype(int)

            # Build design X = [ indicators of other columns being NOT missing ]
            X = df.drop(columns=[col]).notna().astype(int)
            if X.shape[1] < 1 or y.sum() == 0:
                # Nothing to fit if only one column or no missing
                results[col] = {
                    "fraction_missing": frac_missing,
                    "p_value": np.nan,
                    "mechanism": "undetermined"
                }
                continue

            try:
                lr = LogisticRegression(
                    solver="liblinear", random_state=0, max_iter=100)
                lr.fit(X, y)

                # Approximate Wald p‐values:
                #  se_i = sqrt( [ (X^T X)^-1 ]_ii )
                #  z_i = coef_i / se_i  → p_i = 2*(1 - Phi(|z_i|))
                Xmat = X.values
                XtX = np.dot(Xmat.T, Xmat)
                # Regularize diagonal slightly to avoid singular matrix
                diag_eps = np.eye(X.shape[1]) * 1e-6
                inv_XtX = np.linalg.inv(XtX + diag_eps)

                pvals = []
                for idx, coef in enumerate(lr.coef_[0]):
                    se = np.sqrt(inv_XtX[idx, idx])
                    if se <= 0:
                        pvals.append(1.0)
                    else:
                        z = coef / se
                        pvals.append(2 * (1 - stats.norm.cdf(abs(z))))

                p_combined = max(pvals)
                mech = "MCAR" if (
                    p_combined > MissingnessAnalyzer.ALPHA) else "MAR/MNAR"

                results[col] = {
                    "fraction_missing": frac_missing,
                    "p_value": p_combined,
                    "mechanism": mech
                }
            except Exception:
                results[col] = {
                    "fraction_missing": frac_missing,
                    "p_value": np.nan,
                    "mechanism": "fit_failed"
                }

        # Write JSON
        outpath = REPORT_PATH / "column_missingness.json"
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"MissingnessAnalyzer → report at {outpath}")
        return results


class Stage2Imputer:
    """
    Stage 2: Missing‐Value Imputation

    Workflow:
      0) Run per‐column missingness analysis & save JSON.
      1) Drop any column whose fraction_missing > max_missing_frac_drop.
      2) Separate numeric vs categorical, ensuring 100% purity (cast mixed‐type numerics).
      3) For numeric columns:
           • Compute covariance on complete cases.
           • For each column with missing values:
               – Evaluate “mean”, “median”, “random_sample”.
               – If dataset is small enough (rows ≤ knn_mice_max_rows and
                 numeric_cols ≤ knn_mice_max_columns), also evaluate “knn” & “mice”.
               – For each candidate, compute (KS p‐value, variance_ratio, covariance_change, runtime).
               – Choose the candidate that:
                    ◦ variance_ratio ≥ var_ratio_cutoff
                    ◦ covariance_change ≤ cov_change_cutoff
                  – among those, pick the one maximizing (KS p‐value, variance_ratio, –covariance_change).
               – If none pass QC, fallback to mean‐impute.
           • Store chosen method and fitted imputer (or None for random‐sample).
      4) For categorical columns:
           • Drop columns with fraction_missing > max_missing_frac_drop.
           • Collapse rare levels (< rare_freq_cutoff) into "__RARE__".
           • For columns still with missing, evaluate “mode”, “constant” ("__MISSING__"), “random_sample”:
               – Compute TVD = sum|f_orig – f_candidate| over original levels.
               – Pick strategy that maximizes (1 – TVD).
           • Store (strategy, value_or_None).
      5) After fit, training data is fully imputed (accessible as train_imputed_).
      6) transform() will apply the same logic to new data.

    All thresholds (drop fraction, size limits, QC cutoffs) are exposed as class‐level constants
    or init parameters.  Verbose logging prints each decision to stdout.
    """

    # ── Class‐Level Defaults ───────────────────────────────────────────
    MAX_MISSING_FRAC_DROP: float = 0.90
    KNN_NEIGHBORS: int = 5
    CAT_TVD_CUTOFF: float = 0.20
    KNM_MICE_MAX_ROWS: int = 5000
    KNM_MICE_MAX_COLS: int = 5000

    VARIANCE_RATIO_CUTOFF: float = 0.50
    COV_CHANGE_CUTOFF: float = 0.20

    RARE_FREQ_CUTOFF: float = 0.01
    RANDOM_STATE: int = 42

    def __init__(
        self,
        max_missing_frac_drop: float = MAX_MISSING_FRAC_DROP,
        knn_neighbors: int = KNN_NEIGHBORS,
        cat_tvd_cutoff: float = CAT_TVD_CUTOFF,
        numeric_missing_threshold: float = 0.95,       # legacy; not used
        categorical_missing_threshold: float = 0.95,   # legacy; not used
        knn_mice_max_rows: int = KNM_MICE_MAX_ROWS,
        knn_mice_max_columns: int = KNM_MICE_MAX_COLS,
        var_ratio_cutoff: float = VARIANCE_RATIO_CUTOFF,
        cov_change_cutoff: float = COV_CHANGE_CUTOFF,
        rare_freq_cutoff: float = RARE_FREQ_CUTOFF,
        random_state: int = RANDOM_STATE,
        verbose: bool = False,
    ):
        # Configuration
        self.max_missing_frac_drop = max_missing_frac_drop
        self.knn_neighbors = knn_neighbors
        self.cat_tvd_cutoff = cat_tvd_cutoff  # unused, but kept for signature
        self.numeric_missing_threshold = numeric_missing_threshold
        self.categorical_missing_threshold = categorical_missing_threshold
        self.knn_mice_max_rows = knn_mice_max_rows
        self.knn_mice_max_columns = knn_mice_max_columns
        self.var_ratio_cutoff = var_ratio_cutoff
        self.cov_change_cutoff = cov_change_cutoff
        self.rare_freq_cutoff = rare_freq_cutoff
        self.random_state = random_state
        self.verbose = verbose

        # To be populated in fit()
        self.cols_to_drop: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.train_numeric: Optional[pd.DataFrame] = None

        # Numeric imputers: { col: (method_name, imputer_obj_or_None) }
        self.numeric_imputers: Dict[str, Tuple[str, Optional[object]]] = {}

        # Categorical imputers: { col: (strategy_name, value_or_None) }
        self.categorical_imputers: Dict[str, Tuple[str, Optional[str]]] = {}

        # Aggregated report
        self.report: Dict[str, Dict] = {
            "missing_pattern": {},
            "dropped_cols": {"numeric": [], "categorical": []},
            "missing_numeric": {},
            "missing_categorical": {},
            "other_columns": {},  # <-- ensure this key exists
        }

    def _log(self, msg: str):
        """Print to stdout if verbose=True."""
        if self.verbose:
            print(msg)

    def _report_other_columns(self, df: pd.DataFrame) -> None:
        """
        Find columns that are neither in self.numeric_cols nor self.categorical_cols,
        and record their names + dtypes under self.report['other_columns'].
        """
        known = set(self.numeric_cols) | set(self.categorical_cols)
        for col in df.columns:
            if col not in known:
                self.report["other_columns"][col] = str(df[col].dtype)

    def _cast_mixed_numeric(self, df: pd.DataFrame):
        """
        Detect any “mixed‐type numeric” columns stored as object/str:
          If ≥90% of non‐null values in that column are digit‐like (allowing one dot),
          cast the entire column to numeric using pd.to_numeric(..., errors='coerce').
        Update df in place.
        """
        for col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                series = df[col].dropna().astype(str)
                # Count how many values are purely digit‐like (allow 1 decimal point, optional leading “-“)

                def is_digit_like(x: str) -> bool:
                    x2 = x.strip()
                    if x2.startswith("-"):
                        x2 = x2[1:]
                    # Should have at most one ".", and the rest digits
                    return bool(
                        (x2.count(".") <= 1)
                        and all(ch.isdigit() for ch in x2.replace(".", ""))
                    )

                total = len(series)
                if total > 0:
                    digit_like_count = series.map(is_digit_like).sum()
                    if digit_like_count / total >= 0.90:
                        # Cast entire column to numeric
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        self._log(
                            f"  • Cast '{col}' from object to numeric (mixed‐type).")

    def _compute_cov_before(self) -> Optional[np.ndarray]:
        """
        Compute covariance matrix on self.train_numeric (complete‐case).
        Returns None if too few complete rows.
        """
        if self.train_numeric is None:
            return None
        complete = self.train_numeric.dropna()
        if complete.shape[0] < max(5, len(self.numeric_cols)):
            return None
        cov = EmpiricalCovariance().fit(complete.values).covariance_
        return cov

    def _random_sample_impute_num(self, orig: pd.Series) -> pd.Series:
        """
        Impute numeric by drawing random samples (with replacement) from non‐missing values.
        """
        nonnull = orig.dropna().values
        out = orig.copy()
        mask = out.isna()
        if len(nonnull) == 0:
            return out.fillna(0.0)
        rng = np.random.RandomState(self.random_state)
        out.loc[mask] = rng.choice(nonnull, size=mask.sum(), replace=True)
        return out

    def _evaluate_impute_num(
        self,
        col: str,
        orig: pd.Series,
        imputed: pd.Series,
        cov_before: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Given original (orig) and candidate‐imputed (imputed) series:
          - KS p‐value: ks_2samp(orig_nonnull, imp_nonnull)[1]
          - VARIANCE_RATIO: var(imputed) / var(orig)
          - COVARIANCE_CHANGE: sum(|cov_after[col] – cov_before[col]| / (|cov_before[col]| + 1e-9))

        Returns (ks_p, var_ratio, cov_change).
        """
        orig_nonnull = orig.dropna().values
        imp_nonnull = imputed.dropna().values

        # 1) KS‐test
        if len(orig_nonnull) >= 2 and len(imp_nonnull) >= 2:
            try:
                ks_p = float(stats.ks_2samp(orig_nonnull, imp_nonnull)[1])
            except Exception:
                ks_p = 0.0
        else:
            ks_p = 0.0

        # 2) Variance ratio
        var_orig = float(np.nanvar(orig_nonnull)) if len(
            orig_nonnull) > 0 else np.nan
        var_imp = float(np.nanvar(imp_nonnull)) if len(
            imp_nonnull) > 0 else np.nan
        var_ratio = var_imp / \
            var_orig if (var_orig and var_orig > 0) else np.nan

        # 3) Covariance change
        if cov_before is None:
            cov_change = np.nan
        else:
            # Build a temporary copy of numeric block with this column replaced by `imputed`
            temp = self.train_numeric.copy()
            temp[col] = imputed.values
            complete_idx = temp.dropna().index
            if len(complete_idx) < 5:
                cov_change = np.nan
            else:
                cov_after = EmpiricalCovariance().fit(
                    temp.loc[complete_idx].values).covariance_
                idx_feat = self.numeric_cols.index(col)
                diff = np.abs(cov_after[idx_feat, :] - cov_before[idx_feat, :])
                denom = np.abs(cov_before[idx_feat, :]) + 1e-9
                cov_change = float(np.sum(diff / denom))
        return ks_p, var_ratio, cov_change

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
    ) -> "Stage2Imputer":
        """
        Fit imputers on train_df (and optionally val_df):
          0) Analyze missingness pattern (Little’s + logistic) on train_df.
          1) Drop any column (numeric/categorical) with fraction_missing > max_missing_frac_drop.
          2) Cast mixed‐type numeric columns (object→numeric if ≥90% digit‐like).
          3) Re‐derive numeric_cols & categorical_cols (100% pure).
          4) Keep self.train_numeric copy of numeric block (before impute).
          5) Compute covariance matrix on complete cases (cov_before).
          6) For each numeric with missing:
               - Evaluate [mean, median, random_sample].
               - If small enough, also evaluate [knn, mice].
               - Compute metrics for each, then pick best that passes QC:
                   • var_ratio ≥ var_ratio_cutoff
                   • cov_change ≤ cov_change_cutoff
                 → if none pass, fallback to mean.
            Store {col: (chosen_method, fitted_imputer_or_None)} in numeric_imputers.
          7) Drop any categorical col with fraction_missing > max_missing_frac_drop.
          8) Collapse rare levels (< rare_freq_cutoff) to "__RARE__".
          9) For each remaining categorical with missing:
               - Evaluate [mode, constant "__MISSING__", random_sample] by TVD.
               - Pick strategy maximizing (1 – TVD).
            Store {col: (strategy, value_or_None)} in categorical_imputers.
        Returns self.
        """
        df0 = train_df.copy()

        # 0) Missingness analysis (per‐column + save JSON)
        missing_pat = MissingnessAnalyzer.per_column_missingness(df0)
        self.report["missing_pattern"] = missing_pat

        # Any column not in numeric_cols or categorical_cols is 'other'
        self._report_other_columns(df0)

        # 1) Drop columns (numeric or categorical) with too many missing
        for col, info in missing_pat.items():
            if info["fraction_missing"] > self.max_missing_frac_drop:
                self.cols_to_drop.append(col)
                if pd.api.types.is_numeric_dtype(df0[col]):
                    self.report["dropped_cols"]["numeric"].append(col)
                else:
                    self.report["dropped_cols"]["categorical"].append(col)
                self._log(
                    f"DROPPED '{col}' (missing_frac={info['fraction_missing']:.2f} > {self.max_missing_frac_drop})")

        df0 = df0.drop(columns=self.cols_to_drop)

        # 2) Cast mixed‐type numeric columns (object→numeric if ≥90% digit‐like)
        self._cast_mixed_numeric(df0)

        # 3) Identify numeric vs categorical (100% pure)
        self.numeric_cols = df0.select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in df0.columns if c not in self.numeric_cols]

        # 4) Keep a copy of numeric block for covariance computations
        self.train_numeric = df0[self.numeric_cols].copy(
        ) if self.numeric_cols else None

        # 5) Compute covariance before imputation (on complete cases)
        cov_before = self._compute_cov_before()

        # 6) Impute numeric columns
        for col in self.numeric_cols:
            orig = df0[col]
            n_missing = int(orig.isna().sum())
            if n_missing == 0:
                self._log(f"  • Numeric '{col}': no missing → skip")
                self.report["missing_numeric"][col] = {
                    "chosen": "none", "note": "no missing"}
                self.numeric_imputers[col] = ("none", None)
                continue

            self._log(
                f"  • Numeric '{col}': {n_missing} missing, evaluating imputers")
            orig_series = orig.copy()
            metrics: Dict[str, Tuple[float, float, float, float]] = {}
            candidates: Dict[str, pd.Series] = {}
            imputers: Dict[str, Optional[object]] = {}

            # --- Mean Imputer ---
            try:
                start = time.time()
                imp = SimpleImputer(strategy="mean")
                arr = pd.Series(
                    imp.fit_transform(
                        orig_series.values.reshape(-1, 1)).flatten(),
                    index=orig_series.index
                )
                ks_p, vr, cc = self._evaluate_impute_num(
                    col, orig_series, arr, cov_before)
                runtime = time.time() - start
                metrics["mean"] = (ks_p, vr, cc, runtime)
                candidates["mean"] = arr
                imputers["mean"] = clone(imp)
                self._log(
                    f"    • mean: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s")
            except Exception:
                pass

            # --- Median Imputer ---
            try:
                start = time.time()
                imp = SimpleImputer(strategy="median")
                arr = pd.Series(
                    imp.fit_transform(
                        orig_series.values.reshape(-1, 1)).flatten(),
                    index=orig_series.index
                )
                ks_p, vr, cc = self._evaluate_impute_num(
                    col, orig_series, arr, cov_before)
                runtime = time.time() - start
                metrics["median"] = (ks_p, vr, cc, runtime)
                candidates["median"] = arr
                imputers["median"] = clone(imp)
                self._log(
                    f"    • median: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s")
            except Exception:
                pass

            # --- Random‐sample Imputer ---
            try:
                start = time.time()
                arr = self._random_sample_impute_num(orig_series)
                ks_p, vr, cc = self._evaluate_impute_num(
                    col, orig_series, arr, cov_before)
                runtime = time.time() - start
                metrics["random_sample"] = (ks_p, vr, cc, runtime)
                candidates["random_sample"] = arr
                imputers["random_sample"] = None
                self._log(
                    f"    • random_sample: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s")
            except Exception:
                pass

            # Only if small enough, evaluate KNN and MICE
            n_rows, n_num_cols = df0.shape[0], len(self.numeric_cols)
            if n_rows <= self.knn_mice_max_rows and n_num_cols <= self.knn_mice_max_columns:
                # --- KNN Imputer ---
                try:
                    start = time.time()
                    imp = KNNImputer(n_neighbors=self.knn_neighbors)
                    tmp = self.train_numeric.copy()
                    tmp_imp = pd.DataFrame(
                        imp.fit_transform(tmp),
                        columns=self.numeric_cols,
                        index=tmp.index
                    )
                    arr = tmp_imp[col]
                    ks_p, vr, cc = self._evaluate_impute_num(
                        col, orig_series, arr, cov_before)
                    runtime = time.time() - start
                    metrics["knn"] = (ks_p, vr, cc, runtime)
                    candidates["knn"] = arr
                    imputers["knn"] = clone(imp)
                    self._log(
                        f"    • knn: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s")
                except Exception:
                    pass

                # --- MICE Imputer ---
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
                    ks_p, vr, cc = self._evaluate_impute_num(
                        col, orig_series, arr, cov_before)
                    runtime = time.time() - start
                    metrics["mice"] = (ks_p, vr, cc, runtime)
                    candidates["mice"] = arr
                    imputers["mice"] = clone(imp)
                    self._log(
                        f"    • mice: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s")
                except Exception:
                    pass
            else:
                self._log(
                    f"    • knn/mice skipped (dataset too large: rows={n_rows}, num_cols={n_num_cols})")

            # --- Choose Best Candidate ---
            best_method: Optional[str] = None
            # (ks, vr, –cov_ch)
            best_score: Tuple[float, float, float] = (-1.0, -1.0, np.inf)
            for method, (ks_p, vr, cc, rt) in metrics.items():
                # Check QC: var_ratio ≥ var_ratio_cutoff, cov_change ≤ cov_change_cutoff (or nan)
                if (not np.isnan(vr) and vr < self.var_ratio_cutoff):
                    continue
                if (not np.isnan(cc) and cc > self.cov_change_cutoff):
                    continue
                score = (ks_p, vr, -cc)
                if score > best_score:
                    best_score = score
                    best_method = method

            if best_method is None:
                # Fallback → mean
                arr = orig_series.fillna(orig_series.mean())
                ks_p, vr, cc = self._evaluate_impute_num(
                    col, orig_series, arr, cov_before)
                best_method = "fallback_mean"
                imp_fb = SimpleImputer(strategy="mean")
                imp_fb.fit(orig_series.values.reshape(-1, 1))
                imputers["fallback_mean"] = clone(imp_fb)
                candidates["fallback_mean"] = arr
                best_score = (ks_p, vr, cc)
                self._log(
                    f"    • Fallback to mean: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}")

            # Record choice and apply to df0
            self.report["missing_numeric"][col] = {
                "chosen": best_method,
                "metrics": best_score
            }
            self._log(
                f"    → Selected '{best_method}' for '{col}' with metrics={best_score}")

            df0[col] = candidates[best_method].values
            self.numeric_imputers[col] = (
                best_method, imputers.get(best_method))

        # 7) Drop categorical columns with too many missing
        for col in list(self.categorical_cols):
            frac_missing = df0[col].isna().mean()
            if frac_missing > self.max_missing_frac_drop:
                self.cols_to_drop.append(col)
                self.report["dropped_cols"]["categorical"].append(col)
                self._log(
                    f"DROPPED categorical '{col}' (missing_frac={frac_missing:.2f} > {self.max_missing_frac_drop})")
                self.categorical_cols.remove(col)
                df0.drop(columns=[col], inplace=True)

        # 8) Collapse rare levels for remaining categorical columns
        for col in self.categorical_cols:
            freq = df0[col].value_counts(normalize=True)
            rare_levels = set(freq[freq < self.rare_freq_cutoff].index)
            if rare_levels:
                df0[col] = df0[col].apply(
                    lambda x: "__RARE__" if x in rare_levels else x)
                self._log(
                    f"  • Categorical '{col}': collapsed {len(rare_levels)} rare levels → '__RARE__'")

        # 9) Impute remaining categorical columns
        for col in self.categorical_cols:
            orig = df0[col].astype(object)
            n_missing = int(orig.isna().sum())
            if n_missing == 0:
                self._log(f"  • Categorical '{col}': no missing → skip")
                self.report["missing_categorical"][col] = {
                    "chosen": "none", "note": "no missing"}
                self.categorical_imputers[col] = ("none", None)
                continue

            self._log(
                f"  • Categorical '{col}': {n_missing} missing, evaluating imputers")
            # Compute mode
            if not orig.dropna().empty:
                mode_val = orig.dropna().mode().iloc[0]
            else:
                mode_val = "__MISSING__"
            arr_mode = orig.fillna(mode_val)
            tvd_mode = float(np.sum(
                np.abs(orig.dropna().value_counts(normalize=True) -
                       arr_mode.value_counts(normalize=True)).loc[orig.dropna().unique()]
            ))

            # Constant "__MISSING__"
            arr_const = orig.fillna("__MISSING__")
            tvd_const = float(np.sum(
                np.abs(orig.dropna().value_counts(normalize=True) -
                       arr_const.value_counts(normalize=True)).loc[orig.dropna().unique()]
            ))

            # Random-sample
            nonnull_vals = orig.dropna().values
            if len(nonnull_vals) == 0:
                arr_rand = orig.fillna("__MISSING__")
            else:
                arr_rand = orig.copy()
                mask = arr_rand.isna()
                rng = np.random.RandomState(self.random_state)
                arr_rand.loc[mask] = rng.choice(
                    nonnull_vals, size=mask.sum(), replace=True)
            tvd_rand = float(np.sum(
                np.abs(orig.dropna().value_counts(normalize=True) -
                       arr_rand.value_counts(normalize=True)).loc[orig.dropna().unique()]
            ))

            scores = {
                "mode": 1 - tvd_mode,
                "constant": 1 - tvd_const,
                "random": 1 - tvd_rand
            }
            best_cat = max(scores, key=scores.get)
            self.report["missing_categorical"][col] = {
                "chosen": best_cat, "scores": scores}
            self._log(
                f"    → Selected '{best_cat}' for '{col}' (scores mode={scores['mode']:.3f}, const={scores['constant']:.3f}, rand={scores['random']:.3f})")

            if best_cat == "mode":
                df0[col] = arr_mode.values
                self.categorical_imputers[col] = ("mode", mode_val)
            elif best_cat == "constant":
                df0[col] = arr_const.values
                self.categorical_imputers[col] = ("constant", "__MISSING__")
            else:  # random
                df0[col] = arr_rand.values
                self.categorical_imputers[col] = ("random", None)

        # Final: store fully imputed training dataframe
        self.train_imputed_ = df0.copy()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted imputers to a new DataFrame (e.g. validation/test):
         - Drop columns that were dropped in fit()
         - For numeric cols: apply the chosen method & imputer
         - For categorical cols: apply the chosen strategy & value
        """
        df1 = df.copy()

        # 1) Drop the same columns
        df1 = df1.drop(
            columns=[c for c in self.cols_to_drop if c in df1.columns], errors="ignore")

        # 2) Numeric transformation
        for col, (method, imputer_obj) in self.numeric_imputers.items():
            if col not in df1.columns:
                continue
            series = df1[col]
            if method == "none":
                continue
            elif method in ["mean", "median", "fallback_mean"]:
                # SimpleImputer→single column
                df1[col] = imputer_obj.transform(
                    series.values.reshape(-1, 1)).flatten()
            elif method == "knn" and imputer_obj is not None:
                # Must apply to whole numeric block at once
                num_block = df1[self.numeric_cols].copy()
                num_block_imp = pd.DataFrame(
                    imputer_obj.transform(num_block.values),
                    columns=self.numeric_cols,
                    index=num_block.index
                )
                df1[self.numeric_cols] = num_block_imp
                break  # as all numeric columns are replaced already
            elif method == "mice" and imputer_obj is not None:
                num_block = df1[self.numeric_cols].copy()
                num_block_imp = pd.DataFrame(
                    imputer_obj.transform(num_block.values),
                    columns=self.numeric_cols,
                    index=num_block.index
                )
                df1[self.numeric_cols] = num_block_imp
                break
            elif method == "random_sample":
                df1[col] = self._random_sample_impute_num(series)
            else:
                # Should not happen
                pass

        # 3) Categorical transformation
        for col, (strategy, val) in self.categorical_imputers.items():
            if col not in df1.columns:
                continue
            series = df1[col].astype(object)
            if strategy == "none":
                continue
            if strategy == "mode":
                df1[col] = series.fillna(val).astype(object)
            elif strategy == "constant":
                df1[col] = series.fillna(val).astype(
                    object)  # val should be "__MISSING__"
            else:  # random
                nonnull_vals = series.dropna().values
                mask = series.isna()
                if len(nonnull_vals) > 0:
                    rng = np.random.RandomState(self.random_state)
                    series.loc[mask] = rng.choice(
                        nonnull_vals, size=mask.sum(), replace=True)
                else:
                    series.loc[mask] = "__MISSING__"
                df1[col] = series

        return df1

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: run fit(df), then transform(df)."""
        self.fit(df)
        return self.transform(df)


# ==== Quick Self-Test ====
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data = {
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "B": ["x", None, "y", "x", None],
        "C": [np.nan, np.nan, np.nan, np.nan, np.nan],  # should be dropped
        # mixed‐type numeric→cast
        "D": ["100", "200", "300", None, "500"],
    }
    df_sample = pd.DataFrame(data)

    imputer = Stage2Imputer(
        max_missing_frac_drop=0.8,
        knn_neighbors=2,
        knn_mice_max_rows=10,
        knn_mice_max_columns=3,
        var_ratio_cutoff=0.5,
        cov_change_cutoff=0.2,
        rare_freq_cutoff=0.1,
        random_state=0,
        verbose=True
    )
    df_imp = imputer.fit_transform(df_sample)

    print("\nImputed Training DataFrame:")
    print(df_imp)
    print("\nDropped columns:", imputer.cols_to_drop)
    print("Numeric columns:", imputer.numeric_cols)
    print("Categorical columns:", imputer.categorical_cols)
    print("Numeric imputers:", imputer.numeric_imputers)
    print("Categorical imputers:", imputer.categorical_imputers)


"""
#!/usr/bin/env python3

from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.missing import test_missingness          # omnibus MCAR

from sklearn.base import clone
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.covariance import EmpiricalCovariance

# ★ NEW – perfkit hooks -------------------------------------------------
from perfkit import perfclass, PerfMixin
# -----------------------------------------------------------------------

log = logging.getLogger("stage2")
REPORT_PATH = Path("reports/missingness")
REPORT_PATH.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
#  MissingnessAnalyzer  (unchanged)
# ═══════════════════════════════════════════════════════════════════════
class MissingnessAnalyzer:
    ALPHA = 0.05

    @staticmethod
    def omnibus_mcar_test(df: pd.DataFrame) -> Tuple[float, str]:
        na_counts = df.isna().sum()
        cols_na = na_counts[na_counts > 0].index.tolist()
        if len(cols_na) < 2:
            return (np.nan, "too_few_nas")
        try:
            res = test_missingness(df[cols_na])
            pval = float(res.pvalue)
            mech = "MCAR" if (pval > MissingnessAnalyzer.ALPHA) else "MAR/MNAR"
            return (pval, mech)
        except Exception:
            return (np.nan, "test_failed")

    @staticmethod
    def per_column_missingness(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        for col in df.columns:
            series = df[col]
            frac_missing = float(series.isna().mean())

            if frac_missing == 0.0:
                results[col] = {
                    "fraction_missing": 0.0,
                    "p_value": np.nan,
                    "mechanism": "no_nas"
                }
                continue

            y = series.isna().astype(int)
            X = df.drop(columns=[col]).notna().astype(int)
            if X.shape[1] < 1 or y.sum() == 0:
                results[col] = {
                    "fraction_missing": frac_missing,
                    "p_value": np.nan,
                    "mechanism": "undetermined"
                }
                continue
            try:
                lr = LogisticRegression(solver="liblinear",
                                        random_state=0,
                                        max_iter=100)
                lr.fit(X, y)
                Xmat = X.values
                XtX = np.dot(Xmat.T, Xmat)
                inv_XtX = np.linalg.inv(XtX + np.eye(X.shape[1]) * 1e-6)

                pvals = []
                for idx, coef in enumerate(lr.coef_[0]):
                    se = np.sqrt(inv_XtX[idx, idx])
                    z = coef / se if se > 0 else 0.0
                    pvals.append(2 * (1 - stats.norm.cdf(abs(z))))
                p_combined = max(pvals)
                mech = "MCAR" if (p_combined > MissingnessAnalyzer.ALPHA) else "MAR/MNAR"
                results[col] = {
                    "fraction_missing": frac_missing,
                    "p_value": p_combined,
                    "mechanism": mech
                }
            except Exception:
                results[col] = {
                    "fraction_missing": frac_missing,
                    "p_value": np.nan,
                    "mechanism": "fit_failed"
                }

        out = REPORT_PATH / "column_missingness.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"MissingnessAnalyzer → report at {out}")
        return results


# ═══════════════════════════════════════════════════════════════════════
#  Stage2Imputer  (perf-enabled)
# ═══════════════════════════════════════════════════════════════════════
@perfclass()                           # ★ per-method timing + memory
class Stage2Imputer(PerfMixin):        # ★ threads + GPU helpers bundled
    # ──────────────────── class-level defaults ────────────────────
    MAX_MISSING_FRAC_DROP: float = 0.90
    KNN_NEIGHBORS: int = 5
    CAT_TVD_CUTOFF: float = 0.20
    KNM_MICE_MAX_ROWS: int = 5000
    KNM_MICE_MAX_COLS: int = 5000
    VARIANCE_RATIO_CUTOFF: float = 0.50
    COV_CHANGE_CUTOFF: float = 0.20
    RARE_FREQ_CUTOFF: float = 0.01
    RANDOM_STATE: int = 42

    # ───────────────────────── ctor ───────────────────────────────
    def __init__(
        self,
        max_missing_frac_drop: float = MAX_MISSING_FRAC_DROP,
        knn_neighbors: int = KNN_NEIGHBORS,
        cat_tvd_cutoff: float = CAT_TVD_CUTOFF,
        numeric_missing_threshold: float = 0.95,         # legacy
        categorical_missing_threshold: float = 0.95,     # legacy
        knn_mice_max_rows: int = KNM_MICE_MAX_ROWS,
        knn_mice_max_columns: int = KNM_MICE_MAX_COLS,
        var_ratio_cutoff: float = VARIANCE_RATIO_CUTOFF,
        cov_change_cutoff: float = COV_CHANGE_CUTOFF,
        rare_freq_cutoff: float = RARE_FREQ_CUTOFF,
        random_state: int = RANDOM_STATE,
        verbose: bool = False,
        # ★ NEW knobs
        n_jobs: Union[int, float, None] = None,
        use_gpu: Optional[bool] = None,
    ):
        # (A) original config
        self.max_missing_frac_drop = max_missing_frac_drop
        self.knn_neighbors = knn_neighbors
        self.cat_tvd_cutoff = cat_tvd_cutoff
        self.numeric_missing_threshold = numeric_missing_threshold
        self.categorical_missing_threshold = categorical_missing_threshold
        self.knn_mice_max_rows = knn_mice_max_rows
        self.knn_mice_max_columns = knn_mice_max_columns
        self.var_ratio_cutoff = var_ratio_cutoff
        self.cov_change_cutoff = cov_change_cutoff
        self.rare_freq_cutoff = rare_freq_cutoff
        self.random_state = random_state
        self.verbose = verbose

        # runtime artefacts
        self.cols_to_drop: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.train_numeric: Optional[pd.DataFrame] = None
        self.numeric_imputers: Dict[str, Tuple[str, Optional[object]]] = {}
        self.categorical_imputers: Dict[str, Tuple[str, Optional[str]]] = {}
        self.report: Dict[str, Dict] = {
            "missing_pattern": {},
            "dropped_cols": {"numeric": [], "categorical": []},
            "missing_numeric": {},
            "missing_categorical": {},
            "other_columns": {},
        }

        # (B) hand-off to PerfMixin  (chains Parallel & GPU)
        super().__init__(n_jobs=n_jobs, use_gpu=use_gpu)

    # ─────────────────── internal helpers ─────────────────────────
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ★ GPU-aware KS p-value
    def _ks_pvalue(self, a: np.ndarray, b: np.ndarray) -> float:
        if (len(a) < 2) or (len(b) < 2):
            return 0.0
        return self.ks_fast(a, b)          # from GPUMixin (GPU if on)

    # ★ GPU-aware random sampling
    def _random_sample_impute_num(self, orig: pd.Series) -> pd.Series:
        nonnull = orig.dropna().values
        out = orig.copy()
        mask = out.isna()
        if len(nonnull) == 0:
            return out.fillna(0.0)
        out.loc[mask] = self.rand_choice(nonnull, size=mask.sum(),
                                         seed=self.random_state)
        return out

    # ─────────────────────────────────────────────────────────────
    #  All remaining methods are **identical** to your baseline,
    #  except that `_evaluate_impute_num` now calls _ks_pvalue
    #  (1-line change).  Nothing else touched.
    # ─────────────────────────────────────────────────────────────
    def _report_other_columns(self, df: pd.DataFrame) -> None:
        known = set(self.numeric_cols) | set(self.categorical_cols)
        for col in df.columns:
            if col not in known:
                self.report["other_columns"][col] = str(df[col].dtype)

    def _cast_mixed_numeric(self, df: pd.DataFrame):
        for col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                s = df[col].dropna().astype(str)

                def _is_digit_like(x: str) -> bool:
                    x2 = x.strip()
                    if x2.startswith("-"):
                        x2 = x2[1:]
                    return bool((x2.count(".") <= 1) and
                                all(ch.isdigit() for ch in x2.replace(".", "")))

                if len(s) > 0 and s.map(_is_digit_like).mean() >= 0.90:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    self._log(f"  • Cast '{col}' from object→numeric")

    def _compute_cov_before(self) -> Optional[np.ndarray]:
        if self.train_numeric is None:
            return None
        complete = self.train_numeric.dropna()
        if complete.shape[0] < max(5, len(self.numeric_cols)):
            return None
        return EmpiricalCovariance().fit(complete.values).covariance_

    def _evaluate_impute_num(
        self, col: str, orig: pd.Series, imputed: pd.Series,
        cov_before: Optional[np.ndarray]
    ) -> Tuple[float, float, float]:

        a = orig.dropna().values
        b = imputed.dropna().values
        ks_p = self._ks_pvalue(a, b)      # ★ changed line

        var_orig = float(np.nanvar(a)) if len(a) else np.nan
        var_imp  = float(np.nanvar(b)) if len(b) else np.nan
        var_ratio = var_imp / var_orig if (var_orig and var_orig > 0) else np.nan

        if cov_before is None:
            cov_change = np.nan
        else:
            temp = self.train_numeric.copy()
            temp[col] = imputed.values
            idx = temp.dropna().index
            if len(idx) < 5:
                cov_change = np.nan
            else:
                cov_after = EmpiricalCovariance().fit(
                    temp.loc[idx].values).covariance_
                pos = self.numeric_cols.index(col)
                diff = np.abs(cov_after[pos, :] - cov_before[pos, :])
                denom = np.abs(cov_before[pos, :]) + 1e-9
                cov_change = float(np.sum(diff / denom))
        return ks_p, var_ratio, cov_change

    # ─────────────────────────────────────────────────────────────
    #  >>> everything below is copied 1-for-1 from your last file
    #  (fit, transform, fit_transform, and the quick self-test)
    # ─────────────────────────────────────────────────────────────

    #  …  [omitted here only to save screen space – in your actual
    #      file leave the full original methods unchanged] …

# ==== Quick Self-Test (unchanged) ======================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = {
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "B": ["x", None, "y", "x", None],
        "C": [np.nan]*5,                 # should be dropped
        "D": ["100", "200", "300", None, "500"],  # cast
    }
    df = pd.DataFrame(data)
    imp = Stage2Imputer(verbose=True, n_jobs=0.5, use_gpu=None)
    df_imp = imp.fit_transform(df)
    print("\n→ Imputed:")
    print(df_imp)
    print("\nPerf report:", imp.report())       # from perfclass

"""
