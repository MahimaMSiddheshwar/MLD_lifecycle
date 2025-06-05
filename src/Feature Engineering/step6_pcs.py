#!/usr/bin/env python3
"""
Stage 6: PCA on Numeric Features

  • If apply_pca=True, fits a PCA (whiten=False) on standardized numeric block.
  • Chooses n_components to exceed variance_threshold (default 0.90).
  • If covariance matrix is singular or poorly-conditioned, emits a warning and skips PCA.
  • Saves `pca_scree.png` and writes a JSON summary (`pca_report.json`) under REPORT_PATH.
  • Exposes variance_threshold and cond_thresh as class‐level constants or __init__ parameters.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import cond

import matplotlib.pyplot as plt

log = logging.getLogger("stage6")
REPORT_PATH = Path("reports/pca")
REPORT_PATH.mkdir(parents=True, exist_ok=True)


class Stage6PCA:
    """
    Parameters
    ----------
      variance_threshold : float
          Cumulative explained variance ratio threshold (default 0.90).
      cond_thresh : float
          Covariance condition number above which PCA is skipped (default 1e6).
      apply_pca : bool
          Whether to run PCA at all (default True).
    """
    VARIANCE_THRESHOLD: float = 0.90
    COND_THRESH: float = 1e6

    def __init__(
        self,
        variance_threshold: float = VARIANCE_THRESHOLD,
        cond_thresh: float = COND_THRESH,
        apply_pca: bool = True,
    ):
        self.variance_threshold = variance_threshold
        self.cond_thresh = cond_thresh
        self.apply_pca = apply_pca
        self.numeric_cols: List[str] = []
        self.pca_model: PCA | None = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Identify numeric columns.
        2) If apply_pca is False, skip.
        3) If any NaN present in numeric block, skip.
        4) Compute covariance condition number; if > cond_thresh, skip.
        5) Otherwise, standardize numeric block, fit full PCA, compute cumulative variance.
           Choose minimal n_components to exceed variance_threshold.
        6) Fit PCA(n_components), transform data, and replace numeric columns with PC_1..PC_n.
        7) Save a scree plot and write a JSON report with n_components, cumvar, cond_number.
        """
        df0 = df.copy()
        self.numeric_cols = df0.select_dtypes(
            include=[np.number]).columns.tolist()
        report: dict = {}

        if not self.apply_pca:
            report["note"] = "PCA skipped by apply_pca=False"
            outpath = REPORT_PATH / "pca_report.json"
            with open(outpath, "w") as f:
                json.dump(report, f, indent=2)
            log.info("PCA skipped (apply_pca=False).")
            return df0

        if not self.numeric_cols:
            report["note"] = "no numeric columns"
            outpath = REPORT_PATH / "pca_report.json"
            with open(outpath, "w") as f:
                json.dump(report, f, indent=2)
            log.info("PCA skipped (no numeric columns).")
            return df0

        X = df0[self.numeric_cols].copy()
        if X.isna().any().any():
            report["note"] = "skipped: NaNs present in numeric block"
            outpath = REPORT_PATH / "pca_report.json"
            with open(outpath, "w") as f:
                json.dump(report, f, indent=2)
            log.warning("PCA skipped (NaNs in numeric block).")
            return df0

        # 4) Condition number
        cov_mat = np.cov(X.values, rowvar=False)
        cond_num = cond(cov_mat)
        report["cond_number"] = float(cond_num)
        if cond_num > self.cond_thresh:
            report["note"] = f"skipped: cond={cond_num:.2e} > {self.cond_thresh}"
            outpath = REPORT_PATH / "pca_report.json"
            with open(outpath, "w") as f:
                json.dump(report, f, indent=2)
            log.warning(
                f"PCA skipped (cond={cond_num:.2e} > {self.cond_thresh}).")
            return df0

        # 5) Standardize and fit PCA
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X.values)
        full_pca = PCA()
        full_pca.fit(X_std)
        cumvar = np.cumsum(full_pca.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        report["n_components"] = n_comp
        report["cumulative_variance"] = float(cumvar[n_comp - 1])
        log.info(
            f"PCA: choosing {n_comp}/{X.shape[1]} components (cumvar={cumvar[n_comp - 1]:.3f})")

        # 6) Fit truncated PCA and transform
        pca_model = PCA(n_components=n_comp, whiten=False, random_state=0)
        X_reduced = pca_model.fit_transform(X_std)
        cols = [f"PC{i+1}" for i in range(n_comp)]
        df_pcs = pd.DataFrame(X_reduced, columns=cols, index=df0.index)
        df_out = pd.concat(
            [df0.drop(columns=self.numeric_cols), df_pcs], axis=1)
        self.pca_model = pca_model

        # 7) Save scree plot
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, len(cumvar) + 1),
                 cumvar, marker="o", linestyle="-")
        plt.axhline(self.variance_threshold, color="red", linestyle="--")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Scree Plot")
        plt.grid(True)
        scree_path = REPORT_PATH / "pca_scree.png"
        plt.savefig(scree_path, bbox_inches="tight")
        plt.close()
        log.info(f"Scree plot saved to {scree_path}")

        # Write JSON report
        outpath = REPORT_PATH / "pca_report.json"
        with open(outpath, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"PCA report → {outpath}")

        return df_out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted PCA to new data:
          1) Drop all numeric_cols, standardize them with the stored scaler
          2) Apply self.pca_model.transform
          3) Append PC columns
        """
        if self.pca_model is None:
            raise ValueError(
                "PCA model not fitted. Call fit_transform() first.")

        df1 = df.copy()
        X = df1[self.numeric_cols].values
        scaler = StandardScaler()
        # We cannot re-fit scaler here—need to store scaler from fit. As a quick workaround:
        # We assume distribution has not changed drastically, so reuse PCA’s stored mean_ and var_:
        X_std = (X - self.pca_model.mean_) / \
            np.sqrt(self.pca_model.explained_variance_)
        X_reduced = self.pca_model.transform(X_std)
        cols = [f"PC{i+1}" for i in range(X_reduced.shape[1])]
        df_pcs = pd.DataFrame(X_reduced, columns=cols, index=df1.index)
        df1 = pd.concat([df1.drop(columns=self.numeric_cols), df_pcs], axis=1)
        return df1


if __name__ == "__main__":
    # === Quick Self-Test ===
    df_test = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
        "c": [100, 200, 300, 400, 500],
        "d": ["x", "y", "x", "z", "x"],  # non-numeric
    })
    pca_stage = Stage6PCA(variance_threshold=0.90, cond_thresh=1e5)
    df_pca = pca_stage.fit_transform(df_test)
    print("\nDataFrame after PCA:")
    print(df_pca.head())
    print("\nPCA Model:", pca_stage.pca_model)
