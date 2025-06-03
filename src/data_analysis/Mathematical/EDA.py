"""
EDA.py  –  Phase-4 unified EDA runner   ·   v0.4
================================================
* Univariate, target-aware stats, class-imbalance plot
* Bivariate: point-biserial / Pearson, Spearman, Cramer-V
* Optional pair-plots + corr heat-map
* Multivariate: VIF, Mardia, PCA scree, Hopkins, Breusch–Pagan
* Leakage sniff  (AUC ≈ 1)
* HTML profile       (--profile)
* Pair-plots / heat  (--pairplots)
Usage:
    python -m Data_Analysis.EDA --target is_churn --profile --pairplots
"""

from __future__ import annotations
import argparse
import json
import warnings
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.stats import diagnostic as sm_diag
import statsmodels.api as sm
import pingouin as pg

# ── optional heavy deps ────────────────────────────────────────
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    try:
        from ydata_profiling import ProfileReport      # type: ignore
        PROFILING_OK = True
    except ModuleNotFoundError:
        PROFILING_OK = False

sns.set(style="whitegrid")

DATA = Path("data/interim/clean.parquet")
OUT = Path("reports/eda")
OUT.mkdir(parents=True, exist_ok=True)
UVA_DIR = OUT/"uva/plots"
BVA_DIR = OUT/"bva/plots"
MVA_DIR = OUT/"mva/plots"
for d in (UVA_DIR, BVA_DIR, MVA_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# helper utils
# ═══════════════════════════════════════════════════════════════
def _savefig(dir_: Path, name: str):
    plt.tight_layout()
    plt.savefig(dir_/name, dpi=110)
    plt.close()


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    tbl = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(tbl, correction=False)[0]
    n = tbl.values.sum()
    return np.sqrt(chi2 / (n * (min(tbl.shape) - 1)))


def hopkins_stat(df: pd.DataFrame, sample_frac=.1):
    return pg.hopkins(df, n=max(1, int(sample_frac * len(df))))


# ═══════════════════════════════════════════════════════════════
def run(mode="all", target=None, profile=False, pairplots=False):
    df = pd.read_parquet(DATA)
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes("object").columns.tolist()

    # ── 0. target distribution & class ratio ──────────────────
    if target and target in df.columns:
        tgt = df[target]
        plt.figure(figsize=(4, 3))
        if tgt.nunique() <= 10:
            tgt.value_counts(normalize=True).plot(kind="bar")
            plt.ylabel("ratio")
            plt.title(f"{target} distribution")
        else:
            sns.histplot(tgt, kde=True)
            plt.title(f"{target} distribution")
        _savefig(OUT, "target_distribution.png")

    # ── 1. Univariate -------------------------------------------------
    if mode in {"all", "uva"}:
        rows = []
        for col in num_cols:
            s = df[col].dropna()
            rows.append(dict(
                feature=col,
                mean=s.mean(), median=s.median(), var=s.var(), std=s.std(),
                skew=s.skew(), kurt=s.kurt(),
                iqr=s.quantile(.75)-s.quantile(.25),
                shapiro_p=stats.shapiro(s.sample(min(5000, len(s))))[1],
                dagostino_p=stats.normaltest(s)[1],
                jb_p=stats.jarque_bera(s)[1]
            ))
            sns.histplot(s, kde=True)
            plt.title(f"{col} hist")
            _savefig(UVA_DIR, f"{col}_hist.png")
            sns.boxplot(x=s)
            plt.title(f"{col} box")
            _savefig(UVA_DIR, f"{col}_box.png")
        pd.DataFrame(rows).to_csv(OUT/"univariate_summary.csv", index=False)

    # ── 2. Bivariate --------------------------------------------------
    if mode in {"all", "bva"}:
        # numeric ↔ numeric correlations
        pair_stats = []
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i+1:]:
                pear_r, pear_p = stats.pearsonr(df[c1], df[c2])
                spear_r, spear_p = stats.spearmanr(df[c1], df[c2])
                pair_stats.append(dict(x=c1, y=c2,
                                       pearson_r=pear_r, pearson_p=pear_p,
                                       spearman_r=spear_r, spearman_p=spear_p))
                if pairplots:
                    g = sns.jointplot(x=c1, y=c2, data=df,
                                      kind="reg", height=3)
                    g.figure.tight_layout()
                    g.figure.savefig(BVA_DIR/f"{c1}_vs_{c2}.png")
                    plt.close(g.figure)
        pd.DataFrame(pair_stats).to_csv(
            OUT/"bivariate_summary.csv", index=False)

        # numeric ↔ target (point-biserial or Pearson)
        if target and target in df.columns:
            results = []
            cls = df[target]
            for col in num_cols:
                if col == target:
                    continue
                if cls.nunique() <= 2:
                    r, p = stats.pointbiserialr(df[col], cls)
                    results.append(
                        {"feature": col, "pointbiserial_r": r, "p": p})
                else:
                    r, p = stats.pearsonr(df[col], cls)
                    results.append({"feature": col, "pearson_r": r, "p": p})
            pd.DataFrame(results).to_csv(
                OUT/"bivariate_numeric_vs_target.csv", index=False)

        # categorical ↔ target  Cramer-V
        if target and target in df.columns and cls.nunique() <= 10:
            cv = []
            for col in cat_cols:
                cv.append(
                    {"feature": col, "cramers_v": cramers_v(df[col], cls)})
            pd.DataFrame(cv).to_csv(OUT/"cramers_v.csv", index=False)

        # pair-plot: correlation heat-map
        if pairplots:
            plt.figure(figsize=(6, 5))
            sns.heatmap(df[num_cols].corr(), annot=False, cmap="vlag")
            _savefig(BVA_DIR, "corr_heatmap.png")

    # ── 3. Leakage sniff ---------------------------------------------
    if mode in {"all", "mva"} and target and df[target].nunique() == 2:
        leaks = []
        for col in num_cols:
            if col == target:
                continue
            try:
                auc = roc_auc_score(df[target], df[col])
                if auc > 0.97 or auc < 0.03:
                    leaks.append((col, float(auc)))
            except ValueError:
                pass
        if leaks:
            json.dump({"suspects": leaks}, open(
                OUT/"leakage_flags.json", "w"), indent=2)
            print("🛑 leakage suspects:", leaks)

    # ── 4. Multivariate diagnostics ----------------------------------
    if mode in {"all", "mva"}:
        mva = {}
        X = sm.add_constant(df[num_cols])
        mva["max_vif"] = float(pd.Series([vif(X.values, i) for i in range(X.shape[1])],
                                         index=X.columns).drop("const").max())
        _, p_mardia, _ = pg.multivariate_normality(df[num_cols], alpha=.05)
        mva["mardia_p"] = float(p_mardia)

        # Hopkins
        mva["hopkins"] = float(hopkins_stat(df[num_cols]))

        # PCA scree
        pca = PCA().fit(df[num_cols])
        evr = np.cumsum(pca.explained_variance_ratio_)
        mva["components_90pct"] = int(np.where(evr > 0.9)[0][0]+1)
        plt.plot(evr)
        plt.axhline(.9, ls="--")
        plt.ylabel("cum. var")
        plt.xlabel("components")
        _savefig(MVA_DIR, "pca_scree.png")

        # Breusch–Pagan on first numeric vs target (if regression-style)
        first = num_cols[0]
        if target and df[target].nunique() > 2:
            lm = sm.OLS(df[target], sm.add_constant(df[first])).fit()
            bp_p = sm_diag.het_breuschpagan(lm.resid, lm.model.exog)[1]
            mva["breusch_pagan_p"] = float(bp_p)

        json.dump(mva, open(OUT/"mva_summary.json", "w"), indent=2)
        # dendrogram
        sns.clustermap(df[num_cols].corr(), cmap="vlag")
        _savefig(MVA_DIR, "corr_dendrogram.png")

    # ── 5. Optional HTML profile -------------------------------------
    if profile:
        if not PROFILING_OK:
            print("Install ydata-profiling for HTML report")
        else:
            ProfileReport(df, title="EDA profile").to_file(OUT/"profile.html")
            print("📝  HTML profile at reports/eda/profile.html")

    print(f"✅  Phase-4 EDA finished → {OUT.relative_to(Path.cwd())}")


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "uva", "bva", "mva"])
    ap.add_argument("--target")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--pairplots", action="store_true",
                    help="heavy pair-plots & heat-map")
    run(**vars(ap.parse_args()))


class ExploratoryDataAnalysis:
    """
    Section 3: Exploratory Data Analysis (EDA).
    Provides methods to summarize data and detect potential issues or insights.
    """

    def __init__(self):
        """Initialize EDA stage."""
        self.report = {}

    def analyze(self, df: pd.DataFrame, target_col: str = None):
        """
        Analyze the dataset and populate a report with summary statistics.
        If target_col is provided, include target distribution and correlations.
        :param df: Input DataFrame.
        :param target_col: Optional target column name for analysis.
        :return: The input DataFrame (unchanged), for pipeline chaining.
        """
        # Basic dataset shape and types
        num_rows, num_cols = df.shape
        dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        self.report['num_rows'] = num_rows
        self.report['num_cols'] = num_cols
        self.report['column_types'] = dtypes

        # Missing value summary
        missing_counts = df.isna().sum().to_dict()
        self.report['missing_values'] = missing_counts

        # Target column analysis if provided
        if target_col:
            if target_col not in df.columns:
                raise ValueError(
                    f"target_col '{target_col}' not in DataFrame columns")
            target = df[target_col]
            # Distribution of target
            if target.nunique() <= 20:
                # likely categorical or discrete
                target_counts = target.value_counts(dropna=False).to_dict()
                self.report['target_distribution'] = target_counts
            else:
                # likely continuous
                self.report['target_mean'] = target.mean()
                self.report['target_std'] = target.std()
                self.report['target_min'] = target.min()
                self.report['target_max'] = target.max()

        # Basic statistics for numeric features
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_dict()
            self.report['numeric_summary'] = stats

        # Categorical feature summary (top categories)
        categorical_cols = [col for col in df.columns if df[col].dtype == object or str(
            df[col].dtype).startswith('category')]
        cat_summary = {}
        for col in categorical_cols:
            top_vals = df[col].value_counts(dropna=False).head(5).to_dict()
            cat_summary[col] = top_vals
        if cat_summary:
            self.report['categorical_summary'] = cat_summary

        # Correlation with target (for numeric features)
        if target_col and target_col in numeric_cols:
            # if target is numeric, compute Pearson correlation for numeric features
            correlations = {}
            for col in numeric_cols:
                if col == target_col:
                    continue
                correlations[col] = df[col].corr(df[target_col])
            self.report['correlation_with_target'] = correlations
        elif target_col:
            # if target is categorical (classification), compute a simple variance ratio for numeric features
            correlations = {}
            target = df[target_col]
            if target.nunique() > 1:
                for col in numeric_cols:
                    # use an ANOVA-like variance ratio as correlation measure
                    if df[col].nunique() > 0:
                        overall_var = np.var(df[col].dropna().values)
                        within_var = 0
                        n_total = 0
                        for val in target.unique():
                            grp = df[target == val][col].values
                            n = len(grp)
                            n_total += n
                            if n > 1:
                                within_var += n * np.var(grp)
                        if n_total > 0:
                            within_var = within_var / n_total
                            corr_ratio = 1 - within_var/overall_var if overall_var != 0 else 0
                        else:
                            corr_ratio = 0
                        correlations[col] = corr_ratio
                self.report['numeric_feature_correlation_ratio_with_target'] = correlations
            # (Categorical vs categorical correlation not implemented for brevity)

        # Return the original DataFrame to allow chaining
        return df
