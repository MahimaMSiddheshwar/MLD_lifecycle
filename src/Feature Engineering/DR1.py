import warnings
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Union
from scipy import stats
from scipy.stats import chi2
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer
)
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
    OrdinalEncoder
)
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


class DataQualityPipeline:

    def _validate_for_dr(self) -> Tuple[np.ndarray, str]:
        """
        Ensure DR is feasible. Check:
          1) No NaN in numeric portion  
          2) Enough samples: n_samples ≥ min_samples_per_feature × n_features  
          3) At least 2 numeric features  
          4) If LDA: target must exist and be categorical
        Returns (X_values, error_message). If error non‐empty, skip DR.
        """
        X = self.df[self.numeric_cols].copy()
        if X.isna().any().any():
            return X.values, "DR aborted: NaNs remain in numeric data."

        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < self.min_samples_per_feature * n_feats:
            return X.values, (
                f"DR aborted: insufficient samples ({n_samples}) for {n_feats} numeric features."
            )

        if self._dr_type == "lda":
            if not self.target or self.target not in self.df.columns:
                return X.values, "LDA aborted: no target column provided."
            if pd.api.types.is_numeric_dtype(self.df[self.target]):
                return X.values, "LDA aborted: target must be categorical for LDA."

        return X.values, ""

    def _apply_pca(self, X: np.ndarray) -> None:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_std)
        explained = np.cumsum(pca.explained_variance_ratio_)
        n_comp = int(np.searchsorted(
            explained, self.pca_variance_threshold) + 1)

        self._dr_model = pca
        self.report["dim_reduction"] = {
            "type": "PCA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
            "explained_variance_ratio": explained.tolist(),
        }
        if self.verbose:
            print(
                f"\n🔍 DR: PCA → {n_comp} components (cumulative var={explained[n_comp-1]:.3f})")

    def _apply_kernel_pca(self, X: np.ndarray) -> None:
        gamma = 1.0 / X.shape[1]
        kpca = KernelPCA(kernel="rbf", gamma=gamma,
                         fit_inverse_transform=False)
        kpca.fit(X)
        self._dr_model = kpca
        self.report["dim_reduction"] = {
            "type": "KernelPCA",
            "params": {"kernel": "rbf", "gamma": gamma},
        }
        if self.verbose:
            print(f"\n🔍 DR: KernelPCA (rbf, gamma={gamma:.3f}) fitted")

    def _apply_tsvd(self, X: np.ndarray) -> None:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        n_comp = min(X_std.shape[0] - 1, X_std.shape[1])
        tsvd = TruncatedSVD(n_components=n_comp)
        tsvd.fit(X_std)
        explained = np.cumsum(tsvd.explained_variance_ratio_)
        chosen = int(np.searchsorted(
            explained, self.pca_variance_threshold) + 1)

        self._dr_model = tsvd
        self.report["dim_reduction"] = {
            "type": "TruncatedSVD",
            "n_original_features": X.shape[1],
            "n_components": chosen,
            "explained_variance_ratio": explained.tolist(),
        }
        if self.verbose:
            print(
                f"\n🔍 DR: TruncatedSVD → {chosen} components (cum var={explained[chosen-1]:.3f})")

    def _apply_lda(self, X: np.ndarray) -> None:
        y = self.df[self.target]
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        n_comp = min(len(np.unique(y)) - 1, X.shape[1])
        self._dr_model = lda
        self.report["dim_reduction"] = {
            "type": "LDA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
        }
        if self.verbose:
            print(
                f"\n🔍 DR: LDA → {n_comp} components (n_classes={len(np.unique(y))})")

    def _apply_dimensionality_reduction(self, X: np.ndarray) -> None:
        """
        Dispatch to the chosen DR method (PCA, KPCA, TSVD, LDA).
        """
        if self._dr_type == "pca":
            self._apply_pca(X)
        elif self._dr_type == "kpca":
            self._apply_kernel_pca(X)
        elif self._dr_type == "tsvd":
            self._apply_tsvd(X)
        elif self._dr_type == "lda":
            self._apply_lda(X)
