# src/Data_Analysis/probabilistic_analysis.py

"""
ProbabilisticAnalysis: a dataset‐agnostic, no‐manually‐named‐columns tool
for advanced probability‐based diagnostics (Phase 4D “Probabilistic Analysis”).
Place this file under src/Data_Analysis/ and run via:

    python -m Data_Analysis.probabilistic_analysis \
          --data data/interim/clean.parquet \
          [--target TARGET_COLUMN] \
          [--impute-method mice|knn|simple] \
          [--quantile-output normal|uniform]

All outputs land under reports/probabilistic/.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import QuantileTransformer
from statsmodels.distributions.empirical_distribution import ECDF
from copulas.multivariate import GaussianMultivariate
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure outputs go to a dedicated folder
REPORT_DIR = Path("reports/probabilistic")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class ProbabilisticAnalysis:
    def __init__(self, df: pd.DataFrame, target: str | None = None):
        """
        df : a pandas DataFrame (already cleaned from Phase 3)
        target : name of the target column (if you want MI, CPT, group‐wise stats, or permutation importance)
        """
        self.df = df.copy().reset_index(drop=True)
        self.target = target if target in df.columns else None
        # imputed_data will hold numeric matrix after imputation
        self.imputed_data: pd.DataFrame | None = None
        # best‐fit distribution results {col: (dist_name, params..., aic, ks_stat, ks_p)}
        self.distributions: dict[str, tuple] = {}
        # entropy per column
        self.entropy: dict[str, float] = {}
        # mutual information per feature w.r.t. target
        self.mi_scores: pd.Series | None = None
        # conditional probability tables {cat_col: DataFrame}
        self.cond_tables: dict[str, pd.DataFrame] = {}
        # PIT‐transformed DataFrame
        self.pit_df: pd.DataFrame | None = None
        # quantile‐transformed DataFrame
        self.quantile_df: pd.DataFrame | None = None
        # Bayesian group comparison table
        self.group_stats: pd.DataFrame | None = None
        # copula model instance
        self.copula_model: GaussianMultivariate | None = None
        # Mahalanobis distance series
        self.md_series: pd.Series | None = None
        # permutation importances
        self.perm_importance_: pd.Series | None = None

    def impute_missing(self, method: str = "mice") -> pd.DataFrame:
        """
        Impute missing values in *all numeric* columns using:
          - mice (IterativeImputer w/ BayesianRidge + posterior sampling)
          - knn (KNNImputer)
          - simple (SimpleImputer with mean)
        Returns a DataFrame of shape (n_rows, n_numeric_columns).
        """
        numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns to impute in DataFrame")

        if method == "mice":
            imputer = IterativeImputer(estimator=BayesianRidge(),
                                       sample_posterior=True,
                                       random_state=0)
        elif method == "knn":
            imputer = KNNImputer()
        elif method == "simple":
            imputer = SimpleImputer(strategy="mean")
        else:
            raise ValueError(f"Invalid impute method '{method}'")

        num_data = self.df[numeric_cols]
        imputed_arr = imputer.fit_transform(num_data)
        self.imputed_data = pd.DataFrame(
            imputed_arr, columns=numeric_cols, index=self.df.index)

        # Write imputed numeric matrix to CSV for auditing
        out_path = REPORT_DIR / "imputed_numeric.csv"
        self.imputed_data.to_csv(out_path, index=False)
        print(f"→ Imputed numeric data written to {out_path}")
        return self.imputed_data

    def fit_all_distributions(self) -> dict[str, tuple]:
        """
        For each numeric column, fit a few parametric distributions
        and rank by AIC. Returns a dict mapping:
          {col_name: (best_dist_name, best_params, best_aic, ks_stat, ks_p)}
        """
        numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        if not numeric_cols:
            return {}

        results: dict[str, tuple] = {}
        candidate_distributions = [
            "norm", "lognorm", "gamma", "beta", "weibull_min"]
        for col in numeric_cols:
            values = self.df[col].dropna().values
            if len(values) < 5:
                # Too few data points to fit complex distributions
                continue

            best = None  # (dist_name, params, aic, ks_stat, ks_p)
            for dist_name in candidate_distributions:
                dist = getattr(stats, dist_name)
                try:
                    # MLE fit
                    params = dist.fit(values)
                    ll = np.sum(dist.logpdf(values, *params))
                    k = len(params)
                    aic = 2 * k - 2 * ll
                    ks_stat, ks_p = stats.kstest(
                        values, dist_name, args=params)
                    candidate = (dist_name, params, aic, ks_stat, ks_p)
                    if best is None or candidate[2] < best[2]:
                        best = candidate
                except Exception:
                    continue

            if best is not None:
                results[col] = best

        # Save distributions to JSON
        dist_out = {}
        for col, (name, params, aic, ks, pval) in results.items():
            dist_out[col] = {
                "best_dist": name,
                "params": [float(x) for x in params],
                "aic": float(aic),
                "ks_stat": float(ks),
                "ks_p": float(pval),
            }
        out_path = REPORT_DIR / "best_fit_distributions.json"
        out_path.write_text(json.dumps(dist_out, indent=2))
        print(f"→ Best‐fit distribution info written to {out_path}")

        self.distributions = results
        return results

    def shannon_entropy(self) -> dict[str, float]:
        """
        Compute Shannon entropy for each column:
          - For categorical or low‐cardinality (<20), use value_counts probabilities.
          - For numeric, discretize into 20 bins and compute entropy of histogram.
        Returns {col_name: entropy_value}.
        """
        ent: dict[str, float] = {}
        for col in self.df.columns:
            series = self.df[col].dropna()
            if series.dtype == object or series.nunique() < 20:
                probs = series.value_counts(normalize=True).values
                ent[col] = float(-np.sum(probs * np.log(probs + 1e-9)))
            else:
                # numeric: bin into 20 bins
                hist, _ = np.histogram(series.values, bins=20, density=True)
                ent[col] = float(stats.entropy(hist + 1e-9))
        # Save to CSV
        ent_df = pd.DataFrame.from_dict(
            ent, orient="index", columns=["shannon_entropy"])
        ent_path = REPORT_DIR / "shannon_entropy.csv"
        ent_df.to_csv(ent_path, index=True)
        print(f"→ Shannon entropy per column written to {ent_path}")

        self.entropy = ent
        return ent

    def mutual_info_scores(self) -> pd.Series | None:
        """
        Compute mutual information (or F‐score) between each feature and the target.
        Returns a pandas Series indexed by feature name.
        """
        if self.target is None:
            print("→ Skipping MI: no target column provided.")
            return None

        # Drop rows where target is missing
        df_nonmiss = self.df[[
            self.target] + [c for c in self.df.columns if c != self.target]].dropna(subset=[self.target])
        y = df_nonmiss[self.target]

        # One‐hot encode all non‐numeric features
        X_encoded = pd.get_dummies(df_nonmiss.drop(
            columns=[self.target]), drop_first=True)
        if X_encoded.shape[1] == 0:
            print(
                "→ No features to compute MI on (all columns are target or non‐numerical).")
            return None

        if y.nunique() <= 10:
            # classification MI
            mi_vals = mutual_info_classif(
                X_encoded, y, discrete_features="auto", random_state=0)
        else:
            # regression MI (we use F‐score as proxy via f_classif)
            f_vals, _ = f_classif(X_encoded, y)
            mi_vals = np.nan_to_num(f_vals, nan=0.0)

        mi_series = pd.Series(
            mi_vals, index=X_encoded.columns).sort_values(ascending=False)
        mi_path = REPORT_DIR / "mutual_info_scores.csv"
        mi_series.to_csv(mi_path, header=["mi_score"])
        print(f"→ Mutual information (or F‐score) written to {mi_path}")

        self.mi_scores = mi_series
        return mi_series

    def conditional_probability_tables(self) -> dict[str, pd.DataFrame]:
        """
        For each categorical column (dtype == object), compute:
            P(target | category_value) as a crosstab normalized by index.
        Returns a dict {cat_col: DataFrame}.
        """
        if self.target is None:
            print("→ Skipping CPT: no target provided.")
            return {}

        cat_cols = self.df.select_dtypes(include="object").columns.tolist()
        tables: dict[str, pd.DataFrame] = {}
        for col in cat_cols:
            ct = pd.crosstab(
                self.df[col], self.df[self.target], normalize="index")
            tables[col] = ct
            # Save each table as CSV
            out_path = REPORT_DIR / f"cpt_{col}.csv"
            ct.to_csv(out_path)
        print(
            f"→ Conditional probability tables (categorical vs. {self.target}) saved under {REPORT_DIR}/cpt_*.csv")

        self.cond_tables = tables
        return tables

    def pit_transform(self) -> pd.DataFrame:
        """
        Perform Probability Integral Transform (PIT) for every numeric column:
        ˆu_i = F_empirical(x_i). Returns a DataFrame of same shape as numeric subset.
        """
        numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        if not numeric_cols:
            print("→ Skipping PIT: no numeric columns.")
            return pd.DataFrame()

        transformed: dict[str, np.ndarray] = {}
        for col in numeric_cols:
            series = self.df[col].dropna().values
            ecdf = ECDF(series)
            # Map entire column (including NaN) through ECDF (NaN remain NaN)
            transformed[col] = self.df[col].map(lambda x: float(
                ecdf(x)) if pd.notna(x) else np.nan).values

        self.pit_df = pd.DataFrame(transformed, index=self.df.index)
        out_path = REPORT_DIR / "pit_transformed.csv"
        self.pit_df.to_csv(out_path, index=False)
        print(f"→ PIT‐transformed numeric data written to {out_path}")
        return self.pit_df

    def quantile_transform(self, output_distribution: str = "normal") -> pd.DataFrame:
        """
        Quantile‐transform all numeric columns to desired output_distribution in {"normal","uniform"}.
        Returns the transformed numeric DataFrame.
        """
        numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        if not numeric_cols:
            print("→ Skipping quantile transform: no numeric columns.")
            return pd.DataFrame()

        qt = QuantileTransformer(
            output_distribution=output_distribution, random_state=0)
        numeric_data = self.df[numeric_cols].values
        transformed = qt.fit_transform(numeric_data)
        self.quantile_df = pd.DataFrame(
            transformed, columns=numeric_cols, index=self.df.index)

        out_path = REPORT_DIR / \
            f"quantile_transformed_{output_distribution}.csv"
        self.quantile_df.to_csv(out_path, index=False)
        print(
            f"→ Quantile‐transformed numeric data ({output_distribution}) written to {out_path}")
        return self.quantile_df

    def bayesian_group_comparison(self, feature: str) -> pd.DataFrame | None:
        """
        For a categorical 'feature' vs. numeric 'target', compute:
          – mean of each group
          – 95 % empirical CI (2.5th and 97.5th percentiles)
        Returns a DataFrame indexed by group value.
        """
        if self.target is None:
            print("→ Skipping Bayesian group comparison: no target provided.")
            return None
        if feature not in self.df.columns or self.df[feature].dtype == np.number:
            print(
                f"→ Skipping Bayesian group comparison: '{feature}' is not a categorical column.")
            return None

        groups = self.df.dropna(subset=[self.target]).groupby(feature)[
            self.target]
        stat_rows = []
        for name, grp in groups:
            values = grp.values
            mu = float(np.mean(values))
            ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            stat_rows.append({"group": name, "mean": mu, "CI_2.5": float(
                ci_low), "CI_97.5": float(ci_high)})

        self.group_stats = pd.DataFrame(stat_rows).set_index("group")
        out_path = REPORT_DIR / f"group_stats_{feature}.csv"
        self.group_stats.to_csv(out_path)
        print(
            f"→ Bayesian group comparison for '{feature}' vs. '{self.target}' written to {out_path}")
        return self.group_stats

    def predictive_intervals(self, feature: str, alpha: float = 0.05) -> tuple[float, float] | None:
        """
        Compute empirical predictive interval (lower, upper) at level alpha for a numeric feature.
        """
        if feature not in self.df.columns or self.df[feature].dtype == object:
            print(
                f"→ Skipping predictive intervals: '{feature}' is not numeric.")
            return None

        values = self.df[feature].dropna().values
        lower = float(np.percentile(values, 100 * (alpha / 2)))
        upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
        return lower, upper

    def copula_modeling(self) -> GaussianMultivariate | None:
        """
        Fit a Gaussian copula on all numeric columns (drop NA rows).
        Returns the fitted model instance (from copulas.multivariate).
        """
        numeric_df = self.df.select_dtypes(include=np.number).dropna(axis=0)
        if numeric_df.shape[1] < 2:
            print("→ Skipping copula modeling: fewer than 2 numeric columns.")
            return None

        model = GaussianMultivariate()
        model.fit(numeric_df)
        # Save the copula model’s parameters as JSON
        cop_out = REPORT_DIR / "copula_params.json"
        cop_out.write_text(json.dumps(model.to_dict(), indent=2))
        print(f"→ Gaussian copula parameters written to {cop_out}")
        self.copula_model = model
        return model

    def qq_pp_plots(self, feature: str) -> None:
        """
        Generate side‑by‑side QQ and PP plots for a numeric feature, using best‐fit distribution.
        Saves figure to reports/probabilistic/qqpp_<feature>.png
        """
        if feature not in self.distributions:
            print(
                f"→ Skipping QQ/PP for '{feature}': no fitted distribution found.")
            return

        dist_name, params, _, _, _ = self.distributions[feature]
        values = self.df[feature].dropna().values

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"QQ Plot ({feature})")

        plt.subplot(1, 2, 2)
        sorted_data = np.sort(values)
        ecdf = ECDF(values)
        plt.plot(sorted_data, ecdf(sorted_data), label="Empirical")
        plt.plot(sorted_data, getattr(stats, dist_name).cdf(sorted_data, *params),
                 label=f"{dist_name} CDF")
        plt.legend()
        plt.title(f"PP Plot ({feature})")

        plt.tight_layout()
        out_path = REPORT_DIR / f"qqpp_{feature}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"→ Saved QQ/PP plot for '{feature}' to {out_path}")

    def detect_outliers(self) -> pd.Series | None:
        """
        Compute Mahalanobis distance for each row (numeric columns only).
        Returns a pandas Series indexed by original index, containing distance.
        """
        numeric_df = self.df.select_dtypes(include=np.number).dropna(axis=0)
        if numeric_df.shape[1] < 2:
            print("→ Skipping Mahalanobis: not enough numeric columns.")
            return None

        data = numeric_df.values
        cov = np.cov(data, rowvar=False)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)

        mean = np.mean(data, axis=0)
        diffs = data - mean
        left = diffs @ inv_cov
        md = np.sqrt(np.sum(left * diffs, axis=1))

        # Create a full‐length Series, mapping distances back to original index
        md_indexed = pd.Series(np.nan, index=self.df.index)
        md_indexed.loc[numeric_df.index] = md
        out_path = REPORT_DIR / "mahalanobis_distances.csv"
        md_indexed.to_csv(out_path, header=["mahalanobis_distance"])
        print(f"→ Mahalanobis distances saved to {out_path}")
        self.md_series = md_indexed
        return md_indexed

    def feature_importance(self) -> pd.Series | None:
        """
        Use a random forest (classification if target has <=10 unique values,
        regression otherwise) to compute permutation importance of every feature.
        Returns a Series {feature: importance}.
        """
        if self.target is None:
            print("→ Skipping feature importance: no target provided.")
            return None
        df_nonmiss = self.df.dropna(subset=[self.target])
        y = df_nonmiss[self.target]
        X = pd.get_dummies(df_nonmiss.drop(
            columns=[self.target]), drop_first=True)
        if X.shape[1] == 0:
            print(
                "→ Skipping feature importance: no features left after one‐hot encoding.")
            return None

        if y.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=100, random_state=0)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=0)

        model.fit(X, y)
        perm = permutation_importance(
            model, X, y, n_repeats=10, random_state=0)
        imp_series = pd.Series(perm.importances_mean,
                               index=X.columns).sort_values(ascending=False)
        out_path = REPORT_DIR / "permutation_importance.csv"
        imp_series.to_csv(out_path, header=["importance"])
        print(f"→ Permutation feature importances written to {out_path}")
        self.perm_importance_ = imp_series
        return imp_series

    def diagnostic_plots(self, feature: str) -> None:
        """
        Plot a histogram with KDE and overlay the fitted PDF (from best‐fit distribution).
        Saves to reports/probabilistic/diagnostic_<feature>.png
        """
        if feature not in self.distributions:
            print(
                f"→ Skipping diagnostic plot for '{feature}': no fitted distribution.")
            return

        values = self.df[feature].dropna().values
        dist_name, params, *_ = self.distributions[feature]
        dist = getattr(stats, dist_name)

        plt.figure(figsize=(10, 5))
        sns.histplot(values, stat="density", kde=True,
                     color="skyblue", label="Empirical")
        x = np.linspace(values.min(), values.max(), 200)
        plt.plot(x, dist.pdf(x, *params), color="darkred",
                 lw=2, label=f"Fitted {dist_name}")
        plt.title(f"Histogram + Fitted PDF for '{feature}'")
        plt.legend()
        plt.tight_layout()
        out_path = REPORT_DIR / f"diagnostic_{feature}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"→ Saved diagnostic plot for '{feature}' to {out_path}")


def _cli():
    parser = argparse.ArgumentParser(
        description="Run probabilistic analysis on a cleaned DataFrame."
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the cleaned/interim parquet file (e.g. data/interim/clean.parquet)."
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="Name of the target column (for MI, CPT, group‐stats, importance)."
    )
    parser.add_argument(
        "--impute-method", type=str, default="mice", choices=["mice", "knn", "simple"],
        help="Imputation method for numeric missing values."
    )
    parser.add_argument(
        "--quantile-output", type=str, default="normal", choices=["normal", "uniform"],
        help="Output distribution for quantile transform."
    )
    args = parser.parse_args()

    # Load dataset
    df = pd.read_parquet(args.data)
    print(f"→ Loaded data from {args.data} (shape={df.shape})")

    pa = ProbabilisticAnalysis(df, target=args.target)

    # 1. Impute missing numeric
    pa.impute_missing(method=args.impute_method)

    # 2. Fit parametric distributions & pick best by AIC
    pa.fit_all_distributions()

    # 3. Shannon entropy
    pa.shannon_entropy()

    # 4. Mutual information (if target provided)
    pa.mutual_info_scores()

    # 5. Conditional probability tables (if target provided)
    pa.conditional_probability_tables()

    # 6. PIT transform
    pa.pit_transform()

    # 7. Quantile transform
    pa.quantile_transform(output_distribution=args.quantile_output)

    # 8. Bayesian group comparison for each categorical if target is numeric
    if args.target:
        # Only run group comparison on categorical columns
        for cat_col in df.select_dtypes(include="object").columns:
            pa.bayesian_group_comparison(cat_col)

    # 9. Fit copula on numeric data
    pa.copula_modeling()

    # 10. Mahalanobis outliers
    pa.detect_outliers()

    # 11. Permutation feature importance (if target provided)
    pa.feature_importance()

    # 12. Generate QQ/PP + diagnostic plots for every numeric column that was fitted
    for col in pa.distributions:
        pa.qq_pp_plots(col)
        pa.diagnostic_plots(col)

    print("✅ Probabilistic analysis complete. Check outputs under 'reports/probabilistic/'.")


if __name__ == "__main__":
    _cli()
