#!/usr/bin/env python3
"""
auto_baseline.py

Automatically fits and evaluates multiple baseline strategies on train/test splits
without any user intervention.

Usage:

    from auto_baseline import AutoBaseline

    # Assume `train_df` and `test_df` are pandas.DataFrames containing your data,
    # and `target_col` is the name of the target column.

    baseline = AutoBaseline(target=target_col, verbose=True)
    results = baseline.run(train_df, test_df)

    # `results` is a dict mapping each baseline name to its metric dict.
    # When verbose=True, metrics are printed as they’re computed.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class AutoBaseline:
    """
    Automatically fit & evaluate several common baselines on (train, test).

    For regression targets (float dtype):
      – 'mean_regressor'
      – 'median_regressor'

    For classification targets (non‐float dtype OR discrete integer):
      – 'most_frequent'
      – 'stratified'
      – 'uniform'

    After fitting each baseline, it computes standard metrics and returns a dict.
    """

    def __init__(self, target: str, verbose: bool = False):
        """
        Parameters
        ----------
        target : str
            Name of the target column.
        verbose : bool
            If True, prints each baseline’s metrics as it’s computed.
        """
        self.target = target
        self.verbose = verbose
        self.results: Dict[str, Dict[str, Any]] = {}

    def run(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Fit all applicable baselines on train_df, evaluate on test_df, and return metrics.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A mapping from baseline_name → {metric_name: metric_value, …}.
        """
        y_train = train_df[self.target]
        y_test = test_df[self.target]

        # Determine if regression (continuous) vs classification
        is_regression = pd.api.types.is_float_dtype(y_train.dtype)

        if is_regression:
            self._run_regression_baselines(train_df, test_df, y_train, y_test)
        else:
            self._run_classification_baselines(
                train_df, test_df, y_train, y_test)

        return self.results

    def _run_regression_baselines(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        """
        Fit DummyRegressor strategies: 'mean' and 'median', then evaluate.
        """
        X_train = train_df.drop(columns=[self.target])
        X_test = test_df.drop(columns=[self.target])

        # 1) Mean regressor
        dr_mean = DummyRegressor(strategy="mean")
        dr_mean.fit(X_train, y_train)
        y_pred_mean = dr_mean.predict(X_test)

        metrics_mean = {
            "type": "mean_regressor",
            "mae": float(mean_absolute_error(y_test, y_pred_mean)),
            "mse": float(mean_squared_error(y_test, y_pred_mean)),
            "r2": float(r2_score(y_test, y_pred_mean))
        }
        self.results["mean_regressor"] = metrics_mean
        if self.verbose:
            print(f"[mean_regressor] MAE={metrics_mean['mae']:.4f}, "
                  f"MSE={metrics_mean['mse']:.4f}, R2={metrics_mean['r2']:.4f}")

        # 2) Median regressor
        dr_med = DummyRegressor(strategy="median")
        dr_med.fit(X_train, y_train)
        y_pred_med = dr_med.predict(X_test)

        metrics_med = {
            "type": "median_regressor",
            "mae": float(mean_absolute_error(y_test, y_pred_med)),
            "mse": float(mean_squared_error(y_test, y_pred_med)),
            "r2": float(r2_score(y_test, y_pred_med))
        }
        self.results["median_regressor"] = metrics_med
        if self.verbose:
            print(f"[median_regressor] MAE={metrics_med['mae']:.4f}, "
                  f"MSE={metrics_med['mse']:.4f}, R2={metrics_med['r2']:.4f}")

    def _run_classification_baselines(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        """
        Fit DummyClassifier strategies: 'most_frequent', 'stratified', 'uniform', then evaluate.
        """
        X_train = train_df.drop(columns=[self.target])
        X_test = test_df.drop(columns=[self.target])

        # 1) Most frequent
        dc_mf = DummyClassifier(strategy="most_frequent", random_state=0)
        dc_mf.fit(X_train, y_train)
        y_pred_mf = dc_mf.predict(X_test)

        metrics_mf = {
            "type": "most_frequent",
            "accuracy": float(accuracy_score(y_test, y_pred_mf)),
            "f1": float(f1_score(y_test, y_pred_mf, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_mf, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_mf, zero_division=0))
        }
        self.results["most_frequent"] = metrics_mf
        if self.verbose:
            print(f"[most_frequent] Acc={metrics_mf['accuracy']:.4f}, "
                  f"F1={metrics_mf['f1']:.4f}, Prec={metrics_mf['precision']:.4f}, Rec={metrics_mf['recall']:.4f}")

        # 2) Stratified
        dc_strat = DummyClassifier(strategy="stratified", random_state=0)
        dc_strat.fit(X_train, y_train)
        y_pred_strat = dc_strat.predict(X_test)

        metrics_strat = {
            "type": "stratified",
            "accuracy": float(accuracy_score(y_test, y_pred_strat)),
            "f1": float(f1_score(y_test, y_pred_strat, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_strat, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_strat, zero_division=0))
        }
        self.results["stratified"] = metrics_strat
        if self.verbose:
            print(f"[stratified] Acc={metrics_strat['accuracy']:.4f}, "
                  f"F1={metrics_strat['f1']:.4f}, Prec={metrics_strat['precision']:.4f}, Rec={metrics_strat['recall']:.4f}")

        # 3) Uniform (random)
        dc_unif = DummyClassifier(strategy="uniform", random_state=0)
        dc_unif.fit(X_train, y_train)
        y_pred_unif = dc_unif.predict(X_test)

        metrics_unif = {
            "type": "uniform",
            "accuracy": float(accuracy_score(y_test, y_pred_unif)),
            "f1": float(f1_score(y_test, y_pred_unif, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_unif, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_unif, zero_division=0))
        }
        self.results["uniform"] = metrics_unif
        if self.verbose:
            print(f"[uniform] Acc={metrics_unif['accuracy']:.4f}, "
                  f"F1={metrics_unif['f1']:.4f}, Prec={metrics_unif['precision']:.4f}, Rec={metrics_unif['recall']:.4f}")


"""
import pandas as pd
from auto_baseline import AutoBaseline

# Suppose you already have train_df and test_df:
# train_df = pd.read_parquet("data/splits/train.parquet")
# test_df  = pd.read_parquet("data/splits/test.parquet")
# And your target column is "price" (a float column).

baseline = AutoBaseline(target="price", verbose=True)
regression_results = baseline.run(train_df, test_df)

# regression_results will be a dict like:
# {
#   "mean_regressor": { "type":"mean_regressor", "mae":..., "mse":..., "r2":... },
#   "median_regressor": { "type":"median_regressor", "mae":..., "mse":..., "r2":... }
# }
baseline = AutoBaseline(target="is_churn", verbose=True)
classification_results = baseline.run(train_df, test_df)
"""
