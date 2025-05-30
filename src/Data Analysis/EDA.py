"""
src/notebooks/eda_full.py
---------------------------------------------------------------
Comprehensive EDA for Phase-4 (univariate ▸ bivariate ▸ multivariate).

Usage:
    python -m notebooks.eda_full
"""

from statsmodels.stats import diagnostic as sm_diag
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.decomposition import PCA
from scipy import stats
import statsmodels.api as sm
import pingouin as pg
import pandera as pa
import matplotlib.pyplot as plt
import os
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")

RAW = Path("data/interim/clean.parquet")
PLOT_DIR = Path("reports/eda")
UVA_DIR = PLOT_DIR / "uva" / "plots"
BVA_DIR = PLOT_DIR / "bva" / "plots"
MVA_DIR = PLOT_DIR / "mva" / "plots"
for d in (UVA_DIR, BVA_DIR, MVA_DIR):
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(RAW)

# ──────────────────────────────────────────────────────────────
# 1. UNIVARIATE  ------------------------------------------------
# ──────────────────────────────────────────────────────────────
uni_stats = []

for col in df.select_dtypes(include=["number"]).columns:
    s = df[col].dropna()
    desc = {
        "var": col,
        "mean": s.mean(),
        "median": s.median(),
        "var_pop": s.var(),
        "std": s.std(),
        "skew": stats.skew(s),
        "kurt": stats.kurtosis(s, fisher=False),
        "iqr": s.quantile(.75) - s.quantile(.25),
        "shapiro_p": stats.shapiro(s.sample(min(5000, len(s))))[1],
        "dagostino_p": stats.normaltest(s)[1],
        "jb_p": stats.jarque_bera(s)[1],
    }
    uni_stats.append(desc)

    # visuals
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    sns.histplot(s, kde=True, ax=ax[0])
    ax[0].set_title(f"{col} hist")
    sns.boxplot(x=s, ax=ax[1])
    ax[1].set_title("box")
    sm.qqplot(s, line="s", ax=ax[2])
    ax[2].set_title("QQ")
    fig.tight_layout()
    fig.savefig(UVA_DIR / f"{col}.png")
    plt.close(fig)

pd.DataFrame(uni_stats).to_csv(
    PLOT_DIR / "univariate_summary.csv", index=False)

# ──────────────────────────────────────────────────────────────
# 2. BIVARIATE  -------------------------------------------------
# ──────────────────────────────────────────────────────────────
num_cols = df.select_dtypes("number").columns.tolist()
pair_stats = []

for i, c1 in enumerate(num_cols):
    for c2 in num_cols[i+1:]:
        pear_r, pear_p = stats.pearsonr(df[c1], df[c2])
        spear_r, spear_p = stats.spearmanr(df[c1], df[c2])
        pair_stats.append({
            "x": c1, "y": c2,
            "pearson_r": pear_r, "pearson_p": pear_p,
            "spearman_r": spear_r, "spearman_p": spear_p
        })
        g = sns.jointplot(x=c1, y=c2, data=df, kind="reg", height=4)
        g.figure.savefig(BVA_DIR / f"{c1}_vs_{c2}.png")
        plt.close(g.figure)

# two-group t (gender example)
if {"gender", "amount"}.issubset(df.columns):
    m, f = df.amount[df.gender == "M"], df.amount[df.gender == "F"]
    t, p = stats.ttest_ind(m, f, equal_var=False)
    print("Welch-t p-value (M vs F):", p)

pd.DataFrame(pair_stats).to_csv(
    PLOT_DIR / "bivariate_summary.csv", index=False)

# correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="vlag")
plt.tight_layout()
plt.savefig(BVA_DIR / "corr_heatmap.png")
plt.close()

# χ² example: plan × churn
if {"plan", "is_churn"}.issubset(df.columns):
    chi2, p_value, dof, expected = stats.chi2_contingency(
        pd.crosstab(df.plan, df.is_churn))
    print("χ² plan × churn p-value:", p_value)


# ──────────────────────────────────────────────────────────────
# 3. MULTIVARIATE  ---------------------------------------------
# ──────────────────────────────────────────────────────────────
mva_info = {}

# VIF
X = sm.add_constant(df[num_cols])
vifs = pd.Series([vif(X.values, i) for i in range(X.shape[1])],
                 index=X.columns)
vifs.to_csv(PLOT_DIR / "vif.csv")
mva_info["max_vif"] = vifs.drop("const").max()

# multivariate normality (Mardia)
_, p_mardia, _ = pg.multivariate_normality(df[num_cols], alpha=.05)
mva_info["mardia_p"] = p_mardia

# PCA scree
pca = PCA().fit(df[num_cols])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(.9, ls="--")
plt.ylabel("Cumulative Explained Var")
plt.xlabel("Components")
plt.tight_layout()
plt.savefig(MVA_DIR / "pca_scree.png")
plt.close()
mva_info["components_90pct"] = int(
    np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)[0][0]+1)

# Hopkins statistic
hop = pg.hopkins(df[num_cols], n=df.shape[0]//10)
mva_info["hopkins"] = hop

# heteroscedasticity example
if {"age", "amount"}.issubset(df.columns):
    lm = sm.OLS(df.amount, sm.add_constant(df.age)).fit()
    bp_p = sm_diag.het_breuschpagan(lm.resid, lm.model.exog)[1]
    mva_info["breusch_pagan_p"] = bp_p

# save manifest
MANIFEST = PLOT_DIR / "mva_summary.json"
MANIFEST.write_text(json.dumps(mva_info, indent=2))
print("Multivariate summary saved to", MANIFEST)

# cluster-tendency dendrogram
sns.clustermap(df[num_cols].corr(), cmap="vlag")
plt.savefig(MVA_DIR / "corr_dendrogram.png")
plt.close()

print("✅  EDA completed.  Plots & CSVs saved under",
      PLOT_DIR.relative_to(Path.cwd()))
