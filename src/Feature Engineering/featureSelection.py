import numpy as np
import pandas as pd
import warnings
from typing import List, Optional, Tuple, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import (
    VarianceThreshold, mutual_info_classif, mutual_info_regression,
    f_classif, SelectFromModel, RFE, SequentialFeatureSelector
)
from sklearn.linear_model import (
    LassoCV, LogisticRegressionCV, RidgeCV
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score
)
from time import time

warnings.filterwarnings("ignore")


class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    An advanced feature‐selection class that:
      1) Infers task (classification vs regression) from target.
      2) Pre‐filters: near‐zero variance, high‐correlation, low MI/F‐score.
      3) Model‐based selection: L1‐regularized or tree‐based.
      4) Recursive feature elimination (RFE).
      5) Sequential feature selection (forward/backward).
      6) Drop‐if‐improves: tests dropping each feature via CV.
      7) Reports all steps and final selected feature list.

    Usage:
        fs = AdvancedFeatureSelector(
            nzv_threshold=1e-5,
            corr_threshold=0.90,
            mi_quantile=0.05,
            time_budget=300,
            cv=5,
            n_jobs=-1
        )
        X_sel = fs.fit_transform(X, y)
        report = fs.report_
    """

    def __init__(
        self,
        nzv_threshold: float = 1e-5,
        corr_threshold: float = 0.95,
        mi_quantile: float = 0.10,
        time_budget: int = 300,
        cv: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.nzv_threshold = nzv_threshold
        self.corr_threshold = corr_threshold
        self.mi_quantile = mi_quantile
        self.time_budget = time_budget
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # To be populated
        self.task_: str = ""
        self.baseline_score_: float = 0.0
        self.selected_features_: List[str] = []
        self.report_: Dict[str, Union[List[str], Dict]] = {}

    def _infer_task(self, y: pd.Series) -> str:
        if y.dtype.kind in "ifu" and y.nunique() > 20:
            return "regression"
        else:
            return "classification"

    def _pre_filter(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        df = X.copy()
        # 1) Near-zero variance
        nzv = VarianceThreshold(threshold=self.nzv_threshold)
        nzv.fit(df)
        keep = df.columns[nzv.get_support()].tolist()
        dropped_nzv = [c for c in df.columns if c not in keep]
        df = df[keep]

        # 2) High-correlation
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop_corr = [c for c in upper.columns if (
            upper[c] >= self.corr_threshold).any()]
        df = df.drop(columns=to_drop_corr, errors="ignore")

        # 3) Mutual information / F‐score
        num = df.select_dtypes(include=np.number).columns.tolist()
        if num:
            if self.task_ == "classification":
                mi = mutual_info_classif(
                    df[num], y, discrete_features='auto', random_state=self.random_state)
            else:
                mi = mutual_info_regression(df[num], y)
            mi_ser = pd.Series(mi, index=num)
            thr = mi_ser.quantile(self.mi_quantile)
            to_drop_mi = mi_ser[mi_ser < thr].index.tolist()
            df = df.drop(columns=to_drop_mi, errors="ignore")
        else:
            to_drop_mi = []

        self.report_["pre_filter"] = {
            "dropped_nzv": dropped_nzv,
            "dropped_corr": to_drop_corr,
            "dropped_low_mi": to_drop_mi
        }
        return df

    def _baseline_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if self.task_ == "classification":
            model = LogisticRegressionCV(
                cv=self.cv, n_jobs=self.n_jobs, random_state=self.random_state).fit(X, y)
            score = np.mean(cross_val_score(model, X, y, cv=self.cv,
                                            scoring="accuracy" if y.nunique() > 2 else "f1", n_jobs=self.n_jobs))
        else:
            model = RidgeCV(cv=self.cv).fit(X, y)
            score = np.mean(cross_val_score(model, X, y, cv=self.cv,
                                            scoring="r2", n_jobs=self.n_jobs))
        return score

    def _model_based_select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        if self.task_ == "classification":
            base = LogisticRegressionCV(
                penalty="l1", solver="saga", cv=self.cv,
                n_jobs=self.n_jobs, random_state=self.random_state
            )
        else:
            base = LassoCV(cv=self.cv, n_jobs=self.n_jobs,
                           random_state=self.random_state)

        sfm = SelectFromModel(base, threshold="median")
        sfm.fit(X, y)
        feats = X.columns[sfm.get_support()].tolist()
        self.report_["model_based"] = {
            "method": "L1‐feature‐selection",
            "selected": feats
        }
        return feats

    def _tree_based_select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        if self.task_ == "classification":
            model = RandomForestClassifier(
                n_estimators=100, n_jobs=self.n_jobs, random_state=self.random_state)
        else:
            model = RandomForestRegressor(
                n_estimators=100, n_jobs=self.n_jobs, random_state=self.random_state)
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        thr = importances.median()
        feats = importances[importances >= thr].index.tolist()
        self.report_["tree_based"] = {
            "method": "RandomForest‐importance",
            "selected": feats
        }
        return feats

    def _rfe_select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        if self.task_ == "classification":
            estimator = LogisticRegressionCV(cv=self.cv, solver="liblinear",
                                             multi_class="ovr", n_jobs=self.n_jobs, random_state=self.random_state)
        else:
            estimator = LassoCV(cv=self.cv, n_jobs=self.n_jobs,
                                random_state=self.random_state)
        n_features_to_select = max(1, int(0.5 * X.shape[1]))
        rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=0.1)
        rfe.fit(X, y)
        feats = X.columns[rfe.support_].tolist()
        self.report_["rfe"] = {
            "method": "RFE",
            "selected": feats
        }
        return feats

    def _sequential_select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        if self.task_ == "classification":
            estimator = LogisticRegressionCV(cv=self.cv, solver="liblinear",
                                             multi_class="ovr", n_jobs=self.n_jobs, random_state=self.random_state)
            cv_split = StratifiedKFold(self.cv)
            scoring = "accuracy"
        else:
            estimator = RidgeCV(cv=self.cv)
            cv_split = KFold(self.cv)
            scoring = "r2"
        sfs = SequentialFeatureSelector(
            estimator,
            n_features_to_select=max(1, int(0.3 * X.shape[1])),
            direction="forward",
            scoring=scoring,
            cv=cv_split,
            n_jobs=self.n_jobs
        )
        sfs.fit(X, y)
        feats = X.columns[sfs.get_support()].tolist()
        self.report_["sequential"] = {
            "method": "SequentialFS",
            "selected": feats
        }
        return feats

    def _drop_if_improves(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        baseline = self.baseline_score_
        to_drop = []
        for feat in list(X.columns):
            X_sub = X.drop(columns=[feat])
            score = self._baseline_score(X_sub, y)
            if score >= baseline:
                to_drop.append(feat)
                if self.verbose:
                    print(
                        f"Dropping '{feat}' improved score: {score:.4f} ≥ {baseline:.4f}")
        sel = [c for c in X.columns if c not in to_drop]
        self.report_["drop_improves"] = {
            "dropped": to_drop
        }
        return sel

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        df = X.copy()
        target = pd.Series(y, name="target") if not isinstance(
            y, pd.Series) else y.rename("target")
        self.task_ = self._infer_task(target)
        # encode classification target
        if self.task_ == "classification" and target.dtype.kind not in "iu":
            target = LabelEncoder().fit_transform(target)
        # record baseline
        self.baseline_score_ = self._baseline_score(df, target)
        # pre‐filter
        df_pf = self._pre_filter(df, target)
        # model‐based
        mb = set(self._model_based_select(df_pf, target))
        tb = set(self._tree_based_select(df_pf, target))
        rfe = set(self._rfe_select(df_pf, target))
        seq = set(self._sequential_select(df_pf, target))
        # consensus: features selected by ≥2 methods
        consensus = [f for f in df_pf.columns if sum([
            f in mb, f in tb, f in rfe, f in seq
        ]) >= 2]
        # drop‐if‐improves
        df_cons = df_pf[consensus]
        final = self._drop_if_improves(df_cons, target)
        self.selected_features_ = final
        self.report_["consensus"] = consensus
        self.report_["final_selected"] = final
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
