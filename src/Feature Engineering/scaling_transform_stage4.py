#!/usr/bin/env python3
"""
Stage 4: Scaling & Conditional Extra Transformation

  • Chooses between StandardScaler, RobustScaler, or MinMaxScaler based on skew/kurtosis.
  • Applies chosen scaler; retains scaler for downstream test‐set use.
  • Checks for “approximate normality” after scaling (Shapiro p > alpha & |skew| < skew_thresh).
      – If “not normal enough,” evaluates Box‐Cox, Yeo‐Johnson, and Quantile→Normal.
      – Chooses transform with highest (p_value, −|skew|) ranking.
      – Reverts to “none” if no transform improves normality.
  • Writes a JSON “transform_report.json” with details per column.
  • Exposes all thresholds at top of class.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer

log = logging.getLogger("stage4")
REPORT_PATH = Path("reports/transforms")
REPORT_PATH.mkdir(parents=True, exist_ok=True)


class Stage4Transform:
    """
    Performs scaling and optional extra transformations.

    Constants
    ---------
      SKETCH_FOR_ROBUST : float
          Any |skew| > this → prefer RobustScaler over Standard (default 1.0).
      KURT_THRESH_FOR_ROBUST : float
          Any |kurtosis| > this → prefer RobustScaler (default 5.0).
      SKETCH_FOR_STANDARD : float
          If ALL |skew| < this → StandardScaler (default 0.5).
      ALPHA_NORMAL : float
          Shapiro p-value cutoff → if p < alpha, not normal (default 0.05).
      SKEW_CUTOFF_POST : float
          If |skew| after scaling < this, treat as “Gaussian enough” (default 0.75).
      TRANSFORM_CANDIDATES : list[str]
          ["none", "boxcox", "yeo", "quantile"].
    """
    SKEW_THRESH_FOR_ROBUST: float = 1.0
    KURT_THRESH_FOR_ROBUST: float = 5.0
    SKEW_THRESH_FOR_STANDARD: float = 0.5
    ALPHA_NORMAL: float = 0.05
    SKEW_CUTOFF_POST: float = 0.75
    TRANSFORM_CANDIDATES: List[str] = ["none", "boxcox", "yeo", "quantile"]

    def __init__(self):
        self.numeric_cols: List[str] = []
        self.scaler_name: str = "none"
        self.scaler_model = None
        # {col: "none"/"boxcox"/"yeo"/"quantile"}
        self.transform_choices: Dict[str, str] = {}
        # holds fitted PowerTransformer or QuantileTransformer
        self.transform_models: Dict[str, object] = {}

    @staticmethod
    def _compute_skew_kurtosis(df: pd.DataFrame, cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        skews = {}
        kurts = {}
        for col in cols:
            arr = df[col].dropna().values
            if len(arr) > 2:
                skews[col] = float(stats.skew(arr))
                kurts[col] = float(stats.kurtosis(arr))
            else:
                skews[col] = 0.0
                kurts[col] = 0.0
        return skews, kurts

    def _choose_scaler(self, df: pd.DataFrame, cols: List[str]) -> str:
        skews, kurts = self._compute_skew_kurtosis(df, cols)
        # If ANY column is very skewed or heavy‐tailed → Robust
        for c in cols:
            if abs(skews[c]) > self.SKEW_THRESH_FOR_ROBUST or abs(kurts[c]) > self.KURT_THRESH_FOR_ROBUST:
                return "RobustScaler"
        # If ALL columns are roughly symmetric → Standard
        if all(abs(skews[c]) < self.SKEW_THRESH_FOR_STANDARD for c in cols):
            return "StandardScaler"
        # Otherwise → MinMax
        return "MinMaxScaler"

    @staticmethod
    def _shapiro_pval(arr: np.ndarray) -> float:
        """
        Compute Shapiro‐Wilk p‐value on arr (subsample to 5000 if needed).
        """
        if len(arr) < 3:
            return 1.0
        sample = arr if arr.size <= 5000 else np.random.choice(
            arr, 5000, replace=False)
        try:
            return float(stats.shapiro(sample)[1])
        except:
            return 0.0

    def _evaluate_extra_transform(self, arr: np.ndarray) -> Tuple[str, Dict[str, Tuple[float, float]]]:
        """
        For a 1D array arr, evaluate:
          – "none" (no change),
          – "boxcox" (if all arr > 0),
          – "yeo",
          – "quantile".
        For each, compute (pval, −|skew|). Return best as the one with max (pval, −|skew|).
        """
        scores: Dict[str, Tuple[float, float]] = {}
        if arr.size < 5 or np.unique(arr).size < 5:
            return "none", {"none": (1.0, 0.0)}

        for method in self.TRANSFORM_CANDIDATES:
            try:
                if method == "none":
                    cand = arr.copy()
                elif method == "boxcox":
                    if np.any(arr <= 0):
                        continue
                    cand, _ = stats.boxcox(arr)
                elif method == "yeo":
                    pt = PowerTransformer(
                        method="yeo-johnson", standardize=True)
                    cand = pt.fit_transform(arr.reshape(-1, 1)).flatten()
                else:  # quantile
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=0)
                    cand = qt.fit_transform(arr.reshape(-1, 1)).flatten()

                pval = self._shapiro_pval(cand)
                skew_abs = abs(float(stats.skew(cand)))
                scores[method] = (pval, -skew_abs)
            except:
                continue

        # Pick the method with the largest (pval, −skew) lex ordering
        best = max(scores, key=lambda m: (scores[m][0], scores[m][1]))
        return best, scores

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Identify numeric columns.
        2) Choose a scaler (Standard/Robust/MinMax) based on skew/kurtosis.
        3) Fit + transform data with scaler.
        4) For each column: if Shapiro pval < ALPHA_NORMAL or |skew| > SKEW_CUTOFF_POST,
           evaluate extra transforms (Box‐Cox, Yeo, Quantile). Pick best.
        5) Apply chosen transform to that column (training only).
        6) Write JSON report with chosen scaler, per‐column skew/kurtosis,
           and per‐column transform choice + scores.
        """
        df0 = df.copy()
        self.numeric_cols = df0.select_dtypes(
            include=[np.number]).columns.tolist()

        report: Dict[str, Dict] = {"scaler": {}, "per_column": {}}

        if not self.numeric_cols:
            # Nothing to do
            report["scaler"]["chosen"] = "none"
            report["scaler"]["reason"] = "no numeric cols"
            outpath = REPORT_PATH / "transform_report.json"
            with open(outpath, "w") as f:
                json.dump(report, f, indent=2)
            log.info(f"Transform report → {outpath}")
            return df0

        # 2) Choose scaler
        scaler_name = self._choose_scaler(df0, self.numeric_cols)
        report["scaler"]["chosen"] = scaler_name

        if scaler_name == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_name == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        df0[self.numeric_cols] = scaler.fit_transform(df0[self.numeric_cols])
        self.scaler_name = scaler_name
        self.scaler_model = scaler
        log.info(f"Applied {scaler_name} to numeric columns.")

        # 3) Evaluate per‐column normality & possibly extra transform
        for col in self.numeric_cols:
            arr = df0[col].dropna().values
            pval_scaled = self._shapiro_pval(arr)
            skew_scaled = abs(float(stats.skew(arr))) if len(arr) > 2 else 0.0

            if (pval_scaled > self.ALPHA_NORMAL) and (skew_scaled < self.SKEW_CUTOFF_POST):
                # Scaling already sufficed
                choice = "none"
                scores = {"none": (pval_scaled, -skew_scaled)}
                report["per_column"][col] = {
                    "chosen": choice, "scores": scores}
                self.transform_choices[col] = "none"
                self.transform_models[col] = None
                log.info(
                    f"'{col}': no extra transform (post‐scale p={pval_scaled:.3f}, skew={skew_scaled:.3f})")
                continue

            # Otherwise, evaluate all candidates
            best, scores = self._evaluate_extra_transform(arr)
            report["per_column"][col] = {"chosen": best, "scores": scores}
            self.transform_choices[col] = best

            if best == "boxcox":
                df0[col], _ = stats.boxcox(df0[col].values)
                self.transform_models[col] = None  # no persistent model needed
                log.info(f"Applied Box‐Cox to '{col}'")
            elif best == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                df0[col] = pt.fit_transform(df0[[col]]).flatten()
                self.transform_models[col] = pt
                log.info(f"Applied Yeo‐Johnson to '{col}'")
            elif best == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal", random_state=0)
                df0[col] = qt.fit_transform(df0[[col]]).flatten()
                self.transform_models[col] = qt
                log.info(f"Applied Quantile→Normal to '{col}'")
            else:
                log.info(
                    f"No transform improved '{col}'; keeping scaled values.")

        # Write JSON report
        outpath = REPORT_PATH / "transform_report.json"
        with open(outpath, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Transform report → {outpath}")

        return df0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scaler and extra transforms to new data:
          1) df[numeric_cols] = scaler.transform(...)
          2) For any col with transform_choices != "none", apply the corresponding transform_model
             or Box‐Cox (if no model stored).
        """
        df1 = df.copy()
        if not self.numeric_cols or self.scaler_model is None:
            return df1

        df1[self.numeric_cols] = self.scaler_model.transform(
            df1[self.numeric_cols])
        for col in self.numeric_cols:
            choice = self.transform_choices.get(col, "none")
            if choice == "none":
                continue
            arr = df1[col].values
            if choice == "boxcox":
                df1[col], _ = stats.boxcox(arr)
            elif choice == "yeo":
                pt: PowerTransformer = self.transform_models[col]
                df1[col] = pt.transform(df1[[col]]).flatten()
            else:  # quantile
                qt: QuantileTransformer = self.transform_models[col]
                df1[col] = qt.transform(df1[[col]]).flatten()

        return df1


if __name__ == "__main__":
    # === Quick Self-Test ===
    df_test = pd.DataFrame({
        "a": [1, 2, 3, 4, 1000, 6, 7, 8, 9, 10],
        "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "c": [5, 5, 5, 5, 5, 5, 5, 5, np.nan, 5],
    })
    transformer = Stage4Transform()
    df_trans = transformer.fit_transform(df_test)
    print("\nTransformed DataFrame:")
    print(df_trans)
    print("\nScaler chosen:", transformer.scaler_name)
    print("Transform choices:", transformer.transform_choices)
