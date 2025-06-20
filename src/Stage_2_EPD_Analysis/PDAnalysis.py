#!/usr/bin/env python3
"""
probabilistic_analysis_v2.py – Probabilistic Analysis with auto vs full modes

Usage:
    python probabilistic_analysis_v2.py --data clean.parquet \
        --outdir reports/probabilistic_v2 \
        --mode auto \
        [--target TARGET_COLUMN] \
        [--quantile-output normal|uniform] \
        [--min-dist-count 50] \
        [--entropy-bins 20] \
        [--jobs -1]
"""

import argparse
import json
from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif
from sklearn.preprocessing import QuantileTransformer
from copulas.multivariate import GaussianMultivariate
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance

from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
from itertools import combinations
from scipy.stats import entropy as kl_entropy


REPORT_DIR = None  # set in main()


class ProbabilisticAnalysis:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str | None = None,
        mode: str = "auto",
        quantile_output: str = "normal",
        min_dist_count: int = 50,
        entropy_bins: int = 20,
        jobs: int = 1,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame")

        if target and target not in df.columns:
            print(
                f"⚠️ Warning: Target '{target}' not found in dataframe. Ignoring.")
            target = None

        self.df = df.copy().reset_index(drop=True)
        self.target = target
        self.mode = mode.lower()
        self.quantile_output = quantile_output
        self.min_dist_count = min_dist_count
        self.entropy_bins = entropy_bins
        self.jobs = jobs

        self.distributions: dict[str, tuple] = {}
        self.entropy: dict[str, float] = {}
        self.mi_scores: pd.Series | None = None
        self.cond_tables: dict[str, pd.DataFrame] = {}
        self.pit_df: pd.DataFrame | None = None
        self.quantile_df: pd.DataFrame | None = None
        self.group_stats: pd.DataFrame | None = None
        self.copula_model = None
        self.perm_importance_: pd.Series | None = None

    @staticmethod
    def _fit_one_distribution(col: str, values: np.ndarray):
        """Helper for parallel distribution fitting on a single column."""

        candidate_distros = [
            "norm",        # Normal (Gaussian) — symmetric bell curve
            "lognorm",     # Log-Normal — positive-skewed data, e.g., income
            "gamma",       # Gamma — positive-only, skewed data rainfall, time
            "beta",        # Beta — bounded between 0 and 1, e.g., proportions, probabilities
            "weibull_min",  # Weibull — versatile, survival/failure analysis
            "expon",       # Exponential — memoryless events, time between Poisson events
            "pareto",      # Pareto — long tail distributions wealth, file sizes
            "t",           # Student's t — ~ normal, but  heavier tails - small samples
            "cauchy",      # Cauchy — very heavy-tailed, no mean/variance, use with caution
            "rayleigh",    # Rayleigh — positive values, magnitude of 2D vectors signal noise
            "triang"       # Triangular — known min/mode/max shape, simple bounded distribution
        ]

        best = None  # (dist_name, params, aic, ks_stat, ks_p)
        for name in candidate_distros:
            dist = getattr(stats, name)
            try:
                if name in {"lognorm", "gamma", "expon", "pareto", "rayleigh"}:
                    if np.any(values <= 0):
                        continue  # requires strictly positive
                elif name in {"beta", "triang"}:
                    if np.any(values < 0) or np.any(values > 1):
                        continue  # must be bounded [0, 1]
                params = dist.fit(values)
                ll = np.sum(dist.logpdf(values, *params))
                k = len(params)
                aic = 2 * k - 2 * ll
                ks_stat, ks_p = stats.kstest(values, name, args=params)
                candidate = (name, params, aic, ks_stat, ks_p)

                if best is None or candidate[2] < best[2]:  # minimize AIC
                    best = candidate
            except Exception:
                continue
            # Add Anderson-Darling test for normality as extra info
        try:
            ad_stat = stats.anderson(values, dist="norm").statistic
        except Exception:
            ad_stat = None
        return col, best, ad_stat

    def fit_all_distributions(self) -> dict[str, tuple]:
        """
        Fit distributions in parallel.
        In 'auto' mode, only columns with >= min_dist_count non-null values are fitted.
        In 'full' mode, all numeric columns (with at least one non-null) are attempted.
        """
        numeric_cols = self.df.select_dtypes(
            include=np.number).columns.tolist()
        # filter by count if auto
        if self.mode == "auto":
            numeric_cols = [
                c for c in numeric_cols
                if self.df[c].count() >= self.min_dist_count
            ]

        # prepare tasks
        tasks = [
            (col, self.df[col].dropna().values)
            for col in numeric_cols
            if len(self.df[col].dropna()) >= 5
        ]

        # parallel execution
        results = Parallel(n_jobs=self.jobs)(
            delayed(self._fit_one_distribution)(col, vals)
            for col, vals in tasks
        )

        # collect best fits
        dist_out = {}
        for col, best, ad_stat in results:
            if best is None:
                continue
            name, params, aic, ks, pval = best
            self.distributions[col] = best
            dist_out[col] = {
                "best_dist": name,
                "params": [float(x) if x is not None else None for x in params],
                "aic": float(aic),
                "ks_stat": float(ks),
                "ks_p": float(pval),
                "anderson_darling": float(ad_stat) if ad_stat is not None else None
            }

        # save JSON
        path = REPORT_DIR / "best_fit_distributions.json"
        path.write_text(json.dumps(dist_out, indent=2))
        print(f"→ Best-fit distributions written to {path}")
        return self.distributions

    def shannon_entropy(self) -> dict[str, float]:
        """
        Compute Shannon entropy per column.
        Numeric columns are binned into 'entropy_bins'; categoricals by value frequencies.
        """
        ent = {}
        for col in self.df.columns:
            vals = self.df[col].dropna()
            if vals.empty:
                continue
            if vals.dtype == object or vals.nunique() < self.entropy_bins:
                probs = vals.value_counts(normalize=True).values
                ent[col] = float(-np.sum(probs * np.log(probs + 1e-9)))
            else:
                hist, _ = np.histogram(
                    vals.values, bins=self.entropy_bins, density=True)
                ent[col] = float(stats.entropy(hist + 1e-9))
        self.entropy = ent
        pd.Series(ent, name="shannon_entropy").to_csv(
            REPORT_DIR / "shannon_entropy.csv"
        )
        print(f"→ Shannon entropy saved to {REPORT_DIR/'shannon_entropy.csv'}")
        return ent

    def cramers_v_matrix(self) -> pd.DataFrame:
        """Pairwise Cramér's V for all object columns (categorical ↔ categorical)."""
        obj_cols = self.df.select_dtypes(include='object').columns
        result = pd.DataFrame(index=obj_cols, columns=obj_cols, dtype=float)

        for col1, col2 in combinations(obj_cols, 2):
            confusion_matrix = pd.crosstab(self.df[col1], self.df[col2])
            chi2 = stats.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            r, k = confusion_matrix.shape
            phi2 = chi2 / n
            v = np.sqrt(phi2 / min(k - 1, r - 1))
            result.loc[col1, col2] = result.loc[col2, col1] = v

        np.fill_diagonal(result.values, 1.0)
        out_path = REPORT_DIR / "cramers_v_matrix.csv"
        result.to_csv(out_path)
        print(f"→ Cramér’s V matrix saved to {out_path}")
        return result

    def theils_u_matrix(self) -> pd.DataFrame:
        """Theil's U matrix (asymmetrical) for all categorical pairs."""
        def theils_u(x, y):
            s_xy = mutual_info_score(x, y)
            hx = stats.entropy(np.bincount(pd.factorize(x)[0]))
            return s_xy / hx if hx != 0 else 1.0

        obj_cols = self.df.select_dtypes(include='object').columns
        result = pd.DataFrame(index=obj_cols, columns=obj_cols, dtype=float)

        for col1 in obj_cols:
            for col2 in obj_cols:
                if col1 == col2:
                    result.loc[col1, col2] = 1.0
                else:
                    result.loc[col1, col2] = theils_u(
                        self.df[col1], self.df[col2])

        result.to_csv(REPORT_DIR / "theils_u_matrix.csv")
        print(
            f"→ Theil’s U matrix saved to {REPORT_DIR/'theils_u_matrix.csv'}")
        return result

    def rank_correlations(self) -> pd.DataFrame:
        """Kendall's Tau and Spearman correlations for numeric pairs."""
        num_cols = self.df.select_dtypes(include=np.number).columns
        tau_df = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)
        spear_df = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)

        for c1, c2 in combinations(num_cols, 2):
            tau, _ = stats.kendalltau(self.df[c1], self.df[c2])
            spear, _ = stats.spearmanr(self.df[c1], self.df[c2])
            tau_df.loc[c1, c2] = tau_df.loc[c2, c1] = tau
            spear_df.loc[c1, c2] = spear_df.loc[c2, c1] = spear

        np.fill_diagonal(tau_df.values, 1.0)
        np.fill_diagonal(spear_df.values, 1.0)
        tau_df.to_csv(REPORT_DIR / "kendall_tau.csv")
        spear_df.to_csv(REPORT_DIR / "spearman_corr.csv")
        print("→ Kendall’s Tau and Spearman correlation matrices saved.")
        return tau_df, spear_df

    def mutual_info_scores(self) -> pd.Series | None:
        """
        Compute MI (classification) or F-score regression between features and target.
        """
        if not self.target:
            print("→ No target: skipping mutual_info_scores()")
            return None

        df_nm = self.df.dropna(subset=[self.target])
        y = df_nm[self.target]
        X = pd.get_dummies(df_nm.drop(columns=[self.target]), drop_first=True)
        if X.empty:
            print("→ No features: skipping mutual_info_scores()")
            return None

        if y.nunique() <= 10:
            mi = mutual_info_classif(X, y, random_state=0)
        else:
            f_vals, _ = f_classif(X, y)
            mi = np.nan_to_num(f_vals, nan=0.0)

        mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        mi_series.to_csv(REPORT_DIR / "mutual_info_scores.csv", header=["mi"])
        print(
            f"→ Mutual info scores saved to {REPORT_DIR/'mutual_info_scores.csv'}")
        self.mi_scores = mi_series
        return mi_series

    def conditional_probability_tables(self) -> dict[str, pd.DataFrame]:
        """
        Build P(target|category) tables for each object dtype column.
        """
        if not self.target:
            print("→ No target: skipping conditional_probability_tables()")
            return {}

        tables = {}
        for col in self.df.select_dtypes(include="object").columns:
            ct = pd.crosstab(
                self.df[col], self.df[self.target], normalize="index"
            )
            tables[col] = ct
            ct.to_csv(REPORT_DIR / f"cpt_{col}.csv")
        print(f"→ CPTs saved to {REPORT_DIR}")
        self.cond_tables = tables
        return tables

    def fit_transform(self) -> pd.DataFrame:
        """
        Probability Integral Transform for each numeric column via ECDF.
        """
        from statsmodels.distributions.empirical_distribution import ECDF
        transformed = {}
        for col in self.df.select_dtypes(include=np.number).columns:
            vals = self.df[col].dropna().values
            if vals.size == 0:
                continue
            ecdf = ECDF(vals)
            transformed[col] = self.df[col].map(
                lambda x: ecdf(x) if pd.notna(x) else np.nan)
        pit_df = pd.DataFrame(transformed, index=self.df.index)
        pit_df.to_csv(REPORT_DIR / "pit_transformed.csv", index=False)
        print(f"→ PIT transform saved to {REPORT_DIR/'pit_transformed.csv'}")
        self.pit_df = pit_df
        return pit_df

    def quantile_transform(self) -> pd.DataFrame:
        """
        Quantile transform numeric columns to the specified output_distribution.
        """
        nums = self.df.select_dtypes(include=np.number).columns
        if nums.empty:
            print("→ No numeric cols: skipping quantile_transform()")
            return pd.DataFrame()

        qt = QuantileTransformer(
            output_distribution=self.quantile_output, random_state=0)
        arr = qt.fit_transform(self.df[nums])
        qdf = pd.DataFrame(arr, columns=nums, index=self.df.index)
        qdf.to_csv(
            REPORT_DIR / f"quantile_transformed_{self.quantile_output}.csv", index=False)
        print(f"→ Quantile transform saved to {REPORT_DIR}")
        self.quantile_df = qdf
        return qdf

    def bayesian_group_comparison(self) -> pd.DataFrame | None:
        """
        For each categorical column vs numeric target, compute group means and 95% CI.
        """
        if not self.target:
            print("→ No target: skipping bayesian_group_comparison()")
            return None

        stats_list = []
        for col in self.df.select_dtypes(include="object").columns:
            grp = self.df.dropna(subset=[self.target]).groupby(col)[
                self.target]
            for name, series in grp:
                arr = series.values
                mu = float(arr.mean())
                ci = np.percentile(arr, [2.5, 97.5])
                stats_list.append({
                    "feature": col,
                    "group": name,
                    "mean": mu,
                    "ci_lower": float(ci[0]),
                    "ci_upper": float(ci[1]),
                })
        gs = pd.DataFrame(stats_list)
        gs.to_csv(REPORT_DIR / "bayesian_group_stats.csv", index=False)
        print(
            f"→ Bayesian group stats saved to {REPORT_DIR/'bayesian_group_stats.csv'}")
        self.group_stats = gs
        return gs

    def copula_modeling(self) -> GaussianMultivariate | None:
        """
        Fit Gaussian copula on all numeric columns (drop NA rows).
        """
        numeric_df = self.df.select_dtypes(include=np.number).dropna()
        if numeric_df.shape[1] < 2:
            print("→ Not enough numeric cols: skipping copula_modeling()")
            return None

        model = GaussianMultivariate()
        model.fit(numeric_df)
        path = REPORT_DIR / "copula_params.json"
        path.write_text(json.dumps(model.to_dict(), indent=2))
        print(f"→ Copula params saved to {path}")
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

    def jensen_shannon_divergence(p, q):
        """Compute JSD between two distributions (safe wrapper)."""
        p, q = np.asarray(p), np.asarray(q)
        p, q = p + 1e-9, q + 1e-9
        m = 0.5 * (p + q)
        return 0.5 * (kl_entropy(p, m) + kl_entropy(q, m))

    def population_stability_index(expected, actual, bins=10):
        """Compute PSI for drift detection between expected and actual."""
        expected_perc, _ = np.histogram(expected, bins=bins, density=True)
        actual_perc, _ = np.histogram(actual, bins=bins, density=True)
        expected_perc = expected_perc + 1e-9
        actual_perc = actual_perc + 1e-9
        return np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))

    def drift_and_divergence_tests(self) -> pd.DataFrame:
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        results = []

        # For now, use first 10% vs rest as base vs current
        split_idx = int(0.1 * len(self.df))
        base = self.df.iloc[:split_idx]
        current = self.df.iloc[split_idx:]

        for col in num_cols:
            base_vals = base[col].dropna().values
            curr_vals = current[col].dropna().values
            if len(base_vals) < 5 or len(curr_vals) < 5:
                continue
            try:
                jsd = jensen_shannon_divergence(
                    np.histogram(base_vals, bins=20, density=True)[0],
                    np.histogram(curr_vals, bins=20, density=True)[0],
                )
                psi = population_stability_index(base_vals, curr_vals, bins=20)
                kl = kl_entropy(
                    np.histogram(base_vals, bins=20, density=True)[0] + 1e-9,
                    np.histogram(curr_vals, bins=20, density=True)[0] + 1e-9,
                )
                results.append({
                    "feature": col,
                    "JSD": jsd,
                    "PSI": psi,
                    "KL_divergence": kl
                })
            except Exception:
                continue

        df_out = pd.DataFrame(results)
        df_out.to_csv(REPORT_DIR / "divergence_tests.csv", index=False)
        print(
            f"→ Drift & divergence tests saved to {REPORT_DIR/'divergence_tests.csv'}")
        return df_out

    def bootstrap_column_ci(self, n_iter=1000, ci_level=95) -> pd.DataFrame:
        """
        For each numeric column, compute bootstrap CI for mean & median.
        """
        num_cols = self.df.select_dtypes(include=np.number).columns
        results = []

        for col in num_cols:
            vals = self.df[col].dropna().values
            if len(vals) < 10:
                continue
            means, medians = [], []
            for _ in range(n_iter):
                sample = np.random.choice(vals, size=len(vals), replace=True)
                means.append(np.mean(sample))
                medians.append(np.median(sample))
            ci_mean = np.percentile(
                means, [(100 - ci_level) / 2, 100 - (100 - ci_level) / 2])
            ci_median = np.percentile(
                medians, [(100 - ci_level) / 2, 100 - (100 - ci_level) / 2])
            results.append({
                "feature": col,
                "mean_lower": ci_mean[0], "mean_upper": ci_mean[1],
                "median_lower": ci_median[0], "median_upper": ci_median[1]
            })

        df_out = pd.DataFrame(results)
        df_out.to_csv(REPORT_DIR / "bootstrap_cis.csv", index=False)
        print("→ Bootstrap confidence intervals saved.")
        return df_out

    def univariate_moments(self) -> pd.DataFrame:
        """Compute skewness and kurtosis for all numeric columns."""
        stats_df = self.df.select_dtypes(include=np.number).apply(
            lambda x: pd.Series({
                "skewness": stats.skew(x.dropna()),
                "kurtosis": stats.kurtosis(x.dropna(), fisher=True)
            })
        )
        stats_df.to_csv(REPORT_DIR / "skew_kurtosis.csv")
        print("→ Skewness & kurtosis saved to skew_kurtosis.csv")
        return stats_df

    def generate_final_report(self) -> dict:
        """
        Creates a complete in-memory summary report of all statistical diagnostics.
        Saves both JSON and CSV versions into REPORT_DIR.

        Uses:
        - self.entropy
        - self.distributions
        - self.pit_df
        - self.copula_model
        - self.group_stats
        - self.mi_scores
        - self.perm_importance_
        - self.skew_kurt_cache (assigned internally)
        - self.drift_results_cache (assigned internally)

        Returns:
            flags: dict of feature → list of tags
        """
        if not hasattr(self, "skew_kurt_cache"):
            self.skew_kurt_cache = self.df.select_dtypes(include=np.number).apply(
                lambda x: pd.Series({
                    "skewness": stats.skew(x.dropna()),
                    "kurtosis": stats.kurtosis(x.dropna(), fisher=True)
                })
            )

        if not hasattr(self, "drift_results_cache"):
            self.drift_results_cache = pd.DataFrame(self.drift_and_divergence_tests()) if callable(
                getattr(self, 'drift_and_divergence_tests', None)) else pd.DataFrame()

        flags = {}
        csv_rows = []

        numeric_cols = self.df.select_dtypes(include=np.number).columns
        entropy = self.entropy or self.shannon_entropy()
        pit_cols = self.pit_df.columns if self.pit_df is not None else []

        for col in self.df.columns:
            f = []

            # Skew & Kurtosis
            if col in self.skew_kurt_cache.index:
                skew = self.skew_kurt_cache.loc[col, "skewness"]
                kurt = self.skew_kurt_cache.loc[col, "kurtosis"]
                if abs(skew) > 2:
                    f.append("extreme_skew")
                elif abs(skew) > 1:
                    f.append("moderate_skew")
                if abs(kurt) > 5:
                    f.append("extreme_kurtosis")
                elif abs(kurt) > 3:
                    f.append("moderate_kurtosis")

            # Entropy
            if col in entropy:
                if entropy[col] < 0.2:
                    f.append("very_low_entropy")
                elif entropy[col] < 1.0:
                    f.append("low_entropy")

            # Missing distribution fit
            if col in numeric_cols and col not in self.distributions:
                f.append("no_best_fit_found")

            # PIT coverage
            if col in numeric_cols and (col not in pit_cols or self.pit_df[col].isnull().mean() > 0.8):
                f.append("missing_pit")

            # Drift
            if col in self.drift_results_cache["feature"].values:
                drift_row = self.drift_results_cache[self.drift_results_cache["feature"] == col].iloc[0]
                if drift_row["PSI"] > 0.3 or drift_row["JSD"] > 0.2:
                    f.append("high_drift")
                elif drift_row["PSI"] > 0.1 or drift_row["JSD"] > 0.1:
                    f.append("moderate_drift")

            # Copula modeling
            if col in numeric_cols and self.copula_model is None:
                f.append("missing_copula")

            flags[col] = f
            csv_rows.append({"feature": col, "flags": ", ".join(f)})

        # Save to disk
        json_path = REPORT_DIR / "final_summary.json"
        csv_path = REPORT_DIR / "final_summary.csv"
        json_path.write_text(json.dumps(flags, indent=2))
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

        print(
            f"✅ Final report written to {json_path.name} and {csv_path.name}")
        return flags

    def run_all(self):
        self.fit_all_distributions()
        self.shannon_entropy()
        self.mutual_info_scores()
        self.conditional_probability_tables()
        self.fit_transform()
        self.quantile_transform()
        self.bayesian_group_comparison()
        self.copula_modeling()
        self.cramers_v_matrix()
        self.theils_u_matrix()
        self.rank_correlations()
        self.bootstrap_column_ci()
        self.univariate_moments()
        self.drift_results_cache = self.drift_and_divergence_tests()
        self.skew_kurt_cache = self.univariate_moments()
        self.generate_final_report()

        # # 11. Permutation feature importance (if target provided)
        # pa.feature_importance()

        # # 12. Generate QQ/PP + diagnostic plots for every numeric column that was fitted
        # for col in pa.distributions:
        #   pa.qq_pp_plots(col)
        #   pa.diagnostic_plots(col)


"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to cleaned parquet file")
    parser.add_argument("--outdir", default="reports/probabilistic_v2",
                        help="Directory for outputs")
    parser.add_argument("--mode", choices=["auto", "full"], default="auto",
                        help="auto: filter by thresholds; full: run every test")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--quantile-output", choices=["normal", "uniform"],
                        default="normal", help="Quantile transform distribution")
    parser.add_argument("--min-dist-count", type=int, default=50,
                        help="Minimum non-null count to fit distributions in auto mode")
    parser.add_argument("--entropy-bins", type=int, default=20,
                        help="Number of bins for numeric entropy in auto mode")
    parser.add_argument("--jobs", type=int, default=cpu_count(),
                        help="Number of parallel jobs for distribution fitting")
    args = parser.parse_args()

    global REPORT_DIR
    REPORT_DIR = Path(args.outdir)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    print(f"→ Loaded {args.data} with shape {df.shape}")

    pa = ProbabilisticAnalysis(
        df=df,
        target=args.target,
        mode=args.mode,
        quantile_output=args.quantile_output,
        min_dist_count=args.min_dist_count,
        entropy_bins=args.entropy_bins,
        jobs=args.jobs,
    )
    pa.run_all()
    print("✅ Probabilistic analysis complete.")


if __name__ == "__main__":
    main()
"""
