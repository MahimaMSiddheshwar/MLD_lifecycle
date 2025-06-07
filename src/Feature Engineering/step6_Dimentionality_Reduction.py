#!/usr/bin/env python3
"""
Dimensionality Reduction Stage – Auto-select technique based on data characteristics.

Automatically chooses among PCA, TruncatedSVD, KernelPCA, or LDA (if classification target)
by inspecting:
  • Presence of a categorical target & sufficient class separation → LDA
  • Covariance condition number → PCA if well‐conditioned, TSVD if ill‐conditioned
  • Fallback to KernelPCA for non-linear structure
Replaces numeric columns with components; writes scree plots and JSON report.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from numpy.linalg import cond

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
REPORT_DIR = Path("reports/dimensionality_reduction")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class DimensionalityReduction:
    """
    Auto-select DR technique and apply:

      • LDA: if target is categorical and suitable.
      • PCA: if covariance matrix well-conditioned.
      • TruncatedSVD: if cov matrix ill-conditioned.
      • KernelPCA: fallback for non-linear structure.

    Outputs:
      - JSON report: REPORT_DIR / "dr_report.json"
      - Scree plots for PCA/TSVD: REPORT_DIR / "{method}_scree.png"
    """

    VARIANCE_THRESHOLD: float = 0.90
    COND_THRESHOLD: float = 1e6
    MIN_SAMPLES_PER_FEATURE: int = 5

    def __init__(
        self,
        target: Optional[str] = None,
        variance_threshold: float = VARIANCE_THRESHOLD,
        cond_threshold: float = COND_THRESHOLD,
        verbose: bool = True,
    ):
        self.target = target
        self.variance_threshold = variance_threshold
        self.cond_threshold = cond_threshold
        self.verbose = verbose

        self.numeric_cols: List[str] = []
        self.dr_model = None
        self.report: dict = {}

    def _validate_for_dr(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[str]]:
        X = df[self.numeric_cols].copy()
        if X.isna().any().any():
            return X.values, "DR aborted: NaNs remain after imputation."
        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < self.MIN_SAMPLES_PER_FEATURE * n_feats:
            return X.values, (
                f"DR aborted: insufficient samples ({n_samples}) "
                f"for features ({n_feats})."
            )
        if self.target and self.target in df.columns:
            y = df[self.target]
            if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
                if y.nunique() < 2:
                    return X.values, "LDA aborted: target has <2 classes."
            else:
                # numeric target → skip LDA
                pass
        return X.values, None

    def _choose_method(self, X: np.ndarray, df: pd.DataFrame) -> str:
        # 1) LDA if classification target present
        if self.target and self.target in df.columns:
            y = df[self.target]
            if (pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)) \
               and y.nunique() >= 2:
                return "lda"

        # 2) Compute covariance condition number
        cov_mat = np.cov(X, rowvar=False)
        cnum = cond(cov_mat)
        self.report["cov_condition_number"] = float(cnum)
        if cnum < self.cond_threshold:
            return "pca"
        else:
            # if more features than samples, prefer TSVD
            n_samples, n_feats = X.shape
            if n_feats > n_samples:
                return "tsvd"
            # otherwise fallback to KernelPCA
            return "kpca"

    def _apply_pca(self, X: np.ndarray):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        full = PCA()
        full.fit(X_std)
        cumvar = np.cumsum(full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, self.variance_threshold) + 1)

        self.dr_model = PCA(n_components=n_comp)
        X_red = self.dr_model.fit_transform(X_std)
        self.report.update({
            "type": "PCA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
            "explained_variance_ratio": cumvar.tolist()
        })

        # scree plot
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(cumvar)+1), cumvar, marker='o')
        plt.axhline(self.variance_threshold, color='red', ls='--')
        plt.xlabel("Components")
        plt.ylabel("Cumulative Variance")
        plt.title("PCA Scree")
        plt.tight_layout()
        plt.savefig(REPORT_DIR/"pca_scree.png")
        plt.close()

        return X_red

    def _apply_tsvd(self, X: np.ndarray):
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        # choose n_comp = min(n_samples-1, n_feats)
        n_samples, n_feats = X_std.shape
        full = TruncatedSVD(n_components=min(
            n_samples-1, n_feats), random_state=0)
        full.fit(X_std)
        cumvar = np.cumsum(full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, self.variance_threshold) + 1)

        self.dr_model = TruncatedSVD(n_components=n_comp, random_state=0)
        X_red = self.dr_model.fit_transform(X_std)
        self.report.update({
            "type": "TruncatedSVD",
            "n_original_features": n_feats,
            "n_components": n_comp,
            "explained_variance_ratio": cumvar.tolist()
        })

        # scree plot
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(cumvar)+1), cumvar, marker='o')
        plt.axhline(self.variance_threshold, color='red', ls='--')
        plt.xlabel("Components")
        plt.ylabel("Cumulative Variance")
        plt.title("TSVD Scree")
        plt.tight_layout()
        plt.savefig(REPORT_DIR/"tsvd_scree.png")
        plt.close()

        return X_red

    def _apply_kpca(self, X: np.ndarray):
        gamma = 1.0 / X.shape[1]
        kp = KernelPCA(kernel="rbf", gamma=gamma,
                       random_state=0, fit_inverse_transform=False)
        X_red = kp.fit_transform(X)
        self.dr_model = kp
        self.report.update({
            "type": "KernelPCA",
            "params": {"kernel": "rbf", "gamma": gamma}
        })
        return X_red

    def _apply_lda(self, X: np.ndarray, df: pd.DataFrame):
        y = df[self.target].astype('category').cat.codes
        lda = LinearDiscriminantAnalysis()
        X_red = lda.fit_transform(X, y)
        n_comp = X_red.shape[1]
        self.dr_model = lda
        self.report.update({
            "type": "LDA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
            "n_classes": int(df[self.target].nunique())
        })
        return X_red

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df0 = df.copy()
        # identify numeric
        self.numeric_cols = df0.select_dtypes(
            include=np.number).columns.tolist()
        if self.target in self.numeric_cols:
            self.numeric_cols.remove(self.target)

        X, err = self._validate_for_dr(df0)
        if err:
            log.warning(err)
            self.report["note"] = err
            Path(REPORT_DIR/"dr_report.json").write_text(json.dumps(self.report, indent=2))
            return df0

        method = self._choose_method(X, df0)
        if self.verbose:
            log.info(f"Auto-selected DR method: {method.upper()}")

        if method == "pca":
            X_red = self._apply_pca(X)
        elif method == "tsvd":
            X_red = self._apply_tsvd(X)
        elif method == "kpca":
            X_red = self._apply_kpca(X)
        elif method == "lda":
            X_red = self._apply_lda(X, df0)
        else:
            raise RuntimeError(f"Unknown DR method: {method}")

        # assemble output DataFrame
        cols = [f"{method.upper()}_{i+1}" for i in range(X_red.shape[1])]
        df_red = pd.DataFrame(X_red, columns=cols, index=df0.index)
        df_out = pd.concat(
            [df0.drop(columns=self.numeric_cols), df_red], axis=1)

        # write report
        Path(REPORT_DIR/"dr_report.json").write_text(json.dumps(self.report, indent=2))
        return df_out


# Example usage:
# dr = DimensionalityReduction(target="label", verbose=True)
# df_reduced = dr.fit_transform(your_dataframe)
# print(dr.report)
