import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from scipy import stats
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
    IterativeImputer,
)
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer
)
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")


class DataQualityPipeline:

    # ───────────────────── Dimensionality Reduction ──────────────────────────

    def _validate_for_dr(self) -> Tuple[np.ndarray, str]:
        """
        Ensure DR is possible:
          1) No NaNs in numeric portion
          2) Enough samples: n_samples ≥ min_samples_per_feature × n_features
          3) Numeric dims ≥ 2
          4) If LDA and classification target, ensure target is categorical
        Returns (X_values, error_message). If error_message non‐empty, skip DR.
        """
        X = self.df[self.numeric_cols].copy()
        if X.isna().any().any():
            return X.values, "DR aborted: NaNs remain after imputation."

        n_samples, n_feats = X.shape
        if n_feats < 2 or n_samples < self.min_samples_per_feature * n_feats:
            return X.values, (
                f"DR aborted: insufficient samples ({n_samples}) for features ({n_feats})."
            )

        # If user requested LDA but no valid classification target, abort
        if self._dr_type == "lda":
            if self.target is None or self.target not in self.df.columns:
                return X.values, "LDA aborted: no target column provided."
            if not pd.api.types.is_categorical_dtype(self.df[self.target]) and not pd.api.types.is_object_dtype(self.df[self.target]):
                return X.values, "LDA aborted: target must be categorical for LDA."

        return X.values, ""

    def _apply_dr(self, X: np.ndarray) -> None:
        """
        Fit the chosen DR model (PCA/KPCA/SVD/LDA), compute:
          – n_components to reach pca_variance_threshold (only for PCA)
          – explained‐variance (if available)
        """
        if self._dr_type == "pca":
            self._apply_pca(X)
        elif self._dr_type == "kpca":
            self._apply_kernel_pca(X)
        elif self._dr_type == "tsvd":
            self._apply_tsvd(X)
        elif self._dr_type == "lda":
            self._apply_lda(X)
        else:
            return

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
                f"\n🔍 DR: PCA chosen {n_comp} comps ({explained[n_comp-1]:.3f} var)")

    def _apply_kernel_pca(self, X: np.ndarray) -> None:
        # We choose RBF kernel with gamma=1/n_features by default
        gamma = 1.0 / X.shape[1]
        kpca = KernelPCA(kernel="rbf", gamma=gamma,
                         fit_inverse_transform=False)
        kpca.fit(X)
        # We cannot easily get explained variance from KernelPCA,
        # so just record that it ran successfully.
        self._dr_model = kpca
        self.report["dim_reduction"] = {
            "type": "KernelPCA",
            "params": {"kernel": "rbf", "gamma": gamma},
        }
        if self.verbose:
            print(f"\n🔍 DR: KernelPCA (rbf gamma={gamma:.3f}) fitted")

    def _apply_tsvd(self, X: np.ndarray) -> None:
        # TruncatedSVD on standardized data
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        # Choose number of components = min(n_samples-1, n_features)
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
                f"\n🔍 DR: TruncatedSVD chosen {chosen} comps ({explained[chosen-1]:.3f} var)")

    def _apply_lda(self, X: np.ndarray) -> None:
        y = self.df[self.target]
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        # LDA n_components = min(n_classes - 1, n_features)
        n_comp = min(len(np.unique(y)) - 1, X.shape[1])
        self._dr_model = lda
        self.report["dim_reduction"] = {
            "type": "LDA",
            "n_original_features": X.shape[1],
            "n_components": n_comp,
        }
        if self.verbose:
            print(
                f"\n🔍 DR: LDA chosen {n_comp} comps (n_classes={len(np.unique(y))})")

    # ───────────────────────── Main Pipeline ─────────────────────────────────

    def run(self, dr_type: Optional[str] = "pca") -> pd.DataFrame:

        if dr_type:
            self._dr_type = dr_type.lower()
            X_vals, error = self._validate_for_dr()
            if error:
                raise RuntimeError(error)
            self._apply_dr(X_vals)

        return self.df.copy()

    def print_report(self) -> None:
        """
        Print a summary of decisions for each step.
        """
        print("\n" + "=" * 80)
        print("✅ DataQualityPipeline Report")
        print("=" * 80)

        # Univariate outliers
        print("\n▶ Univariate Outliers:")
        for col, methods_dict in self.report["univariate_outliers"].items():
            line = "  • " + col + ": " + ", ".join(
                f"{m}={count}" for m, count in methods_dict.items() if count is not None
            )
            print(line if methods_dict else f"  • {col}: None")

        # Multivariate outliers
        print("\n▶ Multivariate Outliers:")
        for method, idxs in self.report["multivariate_outliers"].items():
            print(f"  • {method}: count={len(idxs)}")

        # Real outliers
        ro = self.report["real_outliers"]
        print(
            f"\n▶ Real Outliers (≥{self.outlier_threshold} flags): count={ro['count']}")

        # Transformation
        print("\n▶ Transformation Summary:")
        for col, info in self.report["transform"].items():
            print(f"  • {col:20s} → {info['chosen']}")

        # Imputation
        print("\n▶ Imputation Summary:")
        for col, info in self.report["imputation"].items():
            ch = info["chosen"]
            m = info["metrics"]
            print(
                f"  • {col:20s} → {ch}, KS_p={m['ks_p']:.3f}, var_ratio={m['var_ratio']:.3f}, cov_change={m['cov_change']:.3f}")

        # Scaler
        print(f"\n▶ Scaler: {self.report['scaler']}")

        # Dimensionality Reduction
        print("\n▶ Dimensionality Reduction:")
        dr = self.report.get("dim_reduction", {})
        if dr:
            dtype = dr["type"]
            nc = dr.get("n_components", None)
            if dtype == "PCA" or dtype == "TruncatedSVD":
                ev = dr["explained_variance_ratio"][nc - 1] if nc else None
                print(
                    f"  • {dtype}: n_original={dr['n_original_features']}, chosen_n={nc}, cumulative_var={ev:.3f}")
            elif dtype == "KernelPCA":
                print(f"  • KernelPCA: parameters={dr['params']}")
            elif dtype == "LDA":
                print(
                    f"  • LDA: n_original={dr['n_original_features']}, chosen_n={nc}")
        else:
            print("  • None")

        # Non‐numeric
        print("\n▶ Non‐numeric columns (unchanged):")
        if self.non_numeric_cols:
            print("  • " + ", ".join(self.non_numeric_cols))
        else:
            print("  • None")

        print("\n" + "=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (as a script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataQualityPipeline Demo")
    parser.add_argument(
        "--input_csv", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--target", required=False, help="Target column (for LDA, etc.)"
    )
    parser.add_argument(
        "--model_family",
        choices=["auto", "linear", "tree", "knn", "svm", "bayesian"],
        default="auto",
        help="Choose pipeline behavior for scaling/imputers",
    )
    parser.add_argument(
        "--dr_type",
        choices=["pca", "kpca", "tsvd", "lda", "none"],
        default="pca",
        help="Which dimensionality reduction to attempt (or none).",
    )
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input_csv)
    pipeline = DataQualityPipeline(
        df=df_raw,
        target_column=args.target,
        model_family=args.model_family,
        verbose=True,
    )

    try:
        df_processed = pipeline.run(
            dr_type=(None if args.dr_type == "none" else args.dr_type))
    except RuntimeError as e:
        print(f"⚠ Dimensionality Reduction skipped: {e}")
    finally:
        pipeline.print_report()

    df_processed.to_csv("processed_output.csv", index=False)
    print("✨ Cleaned data written to 'processed_output.csv'")
