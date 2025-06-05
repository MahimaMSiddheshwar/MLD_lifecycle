#!/usr/bin/env python3
"""
Stage 3: Univariate & Multivariate Outlier Detection

  • Univariate: IQR, Z‐Score, Modified Z (MAD)
  • Multivariate: Mahalanobis on numeric block (if enough complete rows)
  • “Voting” scheme: each method that flags a row gives it 1 vote.
    Any row with votes ≥ vote_threshold is considered “real outlier.”
  • Winsorize flagged “extreme values” (but do NOT drop rows unless 
    multivariate voting also marks it).
  • Exposes all thresholds at top of class.
  • Generates:
      - `outlier_report.json` listing votes per row/column
      - Winsorized DataFrame (no row‐drops by default, since we prefer soft‐treatment)
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict

import scipy.stats as stats
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2

log = logging.getLogger("stage3")
REPORT_PATH = Path("reports/outliers")
REPORT_PATH.mkdir(parents=True, exist_ok=True)


class Stage3Outliers:
    """
    Detects and treats outliers.

    Constants
    ---------
      IQR_FACTOR : float
          Multiplier for IQR method (default 1.5).
      ZSCORE_CUTOFF : float
          |z| cutoff to flag (default 3.0).
      MODZ_CUTOFF : float
          |(x−median)/MAD| cutoff (default 3.5).
      MULTI_VOTE_THRESHOLD : int
          Number of method “votes” to consider row a “real outlier” (default 2).
      MULTI_ALPHA : float
          Significance for Mahalanobis χ² threshold (default 0.975).
      WINSORIZE_FRAC : float
          Fraction to winsorize at each tail (default 0.01).
    """
    IQR_FACTOR: float = 1.5
    ZSCORE_CUTOFF: float = 3.0
    MODZ_CUTOFF: float = 3.5
    MULTI_VOTE_THRESHOLD: int = 2
    MULTI_ALPHA: float = 0.975
    WINSORIZE_FRAC: float = 0.01

    def __init__(self):
        self.numeric_cols: List[str] = []
        self.univ_votes: Dict[int, int] = {}
        self.multiv_votes: List[int] = []
        self.outlier_indices: List[int] = []

    @staticmethod
    def _iqr_flags(series: pd.Series, factor: float) -> List[int]:
        arr = series.dropna().values
        if len(arr) == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - factor * iqr, q3 + factor * iqr
        return series[(series < lb) | (series > ub)].index.tolist()

    @staticmethod
    def _zscore_flags(series: pd.Series, cutoff: float) -> List[int]:
        arr = series.dropna().values
        if len(arr) < 2:
            return []
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        if sigma == 0:
            return []
        z = (series - mu) / sigma
        return series[np.abs(z) > cutoff].index.tolist()

    @staticmethod
    def _modz_flags(series: pd.Series, cutoff: float) -> List[int]:
        arr = series.dropna().values
        if len(arr) < 2:
            return []
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0:
            return []
        modz = 0.6745 * (arr - med) / mad
        return [idx for idx, val in zip(series.dropna().index, modz) if abs(val) > cutoff]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Identify numeric columns.
        2) Run univariate flags (IQR, Z‐Score, Modified Z) and tally votes.
        3) If enough complete rows, run Mahalanobis on numeric block → add to votes.
        4) Mark “real outliers” = votes ≥ MULTI_VOTE_THRESHOLD → drop or winsorize.
        5) By default, we WINSORIZE numeric columns at WINSORIZE_FRAC for any row with votes > 0,
           but do NOT drop the row. If you prefer to drop rows, call `drop_outliers=True`.
        """
        df0 = df.copy()
        self.numeric_cols = df0.select_dtypes(
            include=[np.number]).columns.tolist()
        n_rows = len(df0)
        votes: Dict[int, int] = {i: 0 for i in range(n_rows)}

        # 2) Univariate flags
        report_univ = {}
        for col in self.numeric_cols:
            report_univ[col] = {}
            s = df0[col]
            idx_iqr = self._iqr_flags(s, self.IQR_FACTOR)
            idx_z = self._zscore_flags(s, self.ZSCORE_CUTOFF)
            idx_modz = self._modz_flags(s, self.MODZ_CUTOFF)

            report_univ[col]["iqr"] = len(idx_iqr)
            report_univ[col]["zscore"] = len(idx_z)
            report_univ[col]["modz"] = len(idx_modz)

            for i in idx_iqr + idx_z + idx_modz:
                votes[i] += 1

            log.info(
                f"Univ‐flags for '{col}': IQR={len(idx_iqr)}, Z={len(idx_z)}, MODZ={len(idx_modz)}")

        # 3) Multivariate flags (Mahalanobis)
        report_multi = {}
        numeric_block = df0[self.numeric_cols].dropna()
        if numeric_block.shape[0] >= max(self.MULTI_VOTE_THRESHOLD * len(self.numeric_cols), 5):
            cov = EmpiricalCovariance().fit(numeric_block.values)
            md = cov.mahalanobis(numeric_block.values)
            thresh = chi2.ppf(self.MULTI_ALPHA, df=len(self.numeric_cols))
            mask = md > thresh
            idxs = numeric_block.index[mask]
            report_multi["mahalanobis"] = len(idxs)
            for i in idxs:
                votes[i] += 1
            log.info(
                f"Mahalanobis flagged {len(idxs)} rows (threshold={thresh:.3f})")
        else:
            report_multi["mahalanobis"] = 0
            log.info("Not enough complete rows for Mahalanobis → skipped")

        # 4) Determine real outliers
        self.outlier_indices = [
            i for i, v in votes.items() if v >= self.MULTI_VOTE_THRESHOLD]
        report_final = {"real_outlier_count": len(
            self.outlier_indices), "indices": self.outlier_indices}
        log.info(
            f"Real outliers (votes ≥ {self.MULTI_VOTE_THRESHOLD}): {len(self.outlier_indices)}")

        # 5) Winsorize all numeric columns at tails for rows with ANY votes > 0
        df_wins = df0.copy()
        if self.WINSORIZE_FRAC > 0 and len(votes) > 0:
            mask_any = [i for i, v in votes.items() if v > 0]
            for col in self.numeric_cols:
                arr = df_wins[col].copy()
                lower = np.nanpercentile(arr, 100 * self.WINSORIZE_FRAC)
                upper = np.nanpercentile(arr, 100 * (1 - self.WINSORIZE_FRAC))
                for i in mask_any:
                    val = arr.iloc[i]
                    if pd.notna(val):
                        if val < lower:
                            arr.iloc[i] = lower
                        elif val > upper:
                            arr.iloc[i] = upper
                df_wins[col] = arr
            log.info(
                f"Winsorized numeric columns at {self.WINSORIZE_FRAC*100:.1f}% tails for flagged rows.")

        # 6) Write outlier report JSON
        outlier_report = {
            "univariate": report_univ,
            "multivariate": report_multi,
            "final": report_final,
        }
        outpath = REPORT_PATH / "outlier_report.json"
        with open(outpath, "w") as f:
            json.dump(outlier_report, f, indent=2)
        log.info(f"Outlier report → {outpath}")

        return df_wins


if __name__ == "__main__":
    # === Quick Self-Test ===
    df_test = pd.DataFrame({
        "x": [1, 2, 1000, 4, 5, -999, 7],
        "y": [10, 20, 30, 40, 50, 60, 70],
        "z": [5, 6, 7, 8, 9, 10, 11]
    })
    outlier_stage = Stage3Outliers()
    df_out = outlier_stage.fit_transform(df_test)
    print("\nWinsorized DataFrame:")
    print(df_out)
    print("Outlier Indices:", outlier_stage.outlier_indices)
