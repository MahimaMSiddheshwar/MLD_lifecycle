"""
src/notebooks/eda_advanced.py
-----------------------------------------------------------------
Extra analyses not in eda_full.py – categorical association,
mutual-info, leakage heuristics, TS decomposition, auto-profilers.
"""

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import mutual_info_classif, mutual_info_regression
from scipy import stats
import matplotlib.pyplot as plt
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks")


try:
    import ydata_profiling as yp            # optional, for HTML report
except ImportError:
    warnings.warn("pip install ydata-profiling to enable full report")

ADV_DIR = Path("reports/eda/advanced")
ADV_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/interim/clean.parquet")

# ──────────────────────────────────────────────────────────────
# 1.  Categorical Association  (Cramer-V matrix)  ---------------
# ──────────────────────────────────────────────────────────────
cat_cols = df.select_dtypes("object").columns


def cramers_v(conf):
    chi2 = stats.chi2_contingency(conf)[0]
    n = conf.sum().sum()
    r, k = conf.shape
    return np.sqrt(chi2 / (n*(min(k-1, r-1))))


V = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
for i in cat_cols:
    for j in cat_cols:
        V.loc[i, j] = cramers_v(pd.crosstab(df[i], df[j]))
sns.heatmap(V, annot=True, cmap="rocket_r")
plt.title("Cramer-V (categorical-categorical strength)")
plt.tight_layout()
plt.savefig(ADV_DIR/"cramers_v.png")
plt.close()

# ──────────────────────────────────────────────────────────────
# 2.  Mutual Information vs target  -----------------------------
# ──────────────────────────────────────────────────────────────
target = "is_churn" if "is_churn" in df else df.columns[-1]
X_num = df.select_dtypes("number").drop(columns=target, errors="ignore")
X_cat = df[cat_cols]

mi_num = mutual_info_classif(X_num, df[target], random_state=0)
mi_cat = mutual_info_classif(
    pd.get_dummies(X_cat, drop_first=True),
    df[target], random_state=0) if len(cat_cols) else []

mi_series = pd.Series(
    np.concatenate([mi_num, mi_cat]),
    index=list(X_num.columns) +
    list(pd.get_dummies(X_cat, drop_first=True).columns),
    name="mutual_info")
mi_series.sort_values(ascending=False).to_csv(ADV_DIR/"mutual_info.csv")

# barplot
mi_series[:20].plot.barh(figsize=(6, 5))
plt.gca().invert_yaxis()
plt.title("Top MI w/ target")
plt.tight_layout()
plt.savefig(ADV_DIR/"mutual_info_top20.png")
plt.close()

# ──────────────────────────────────────────────────────────────
# 3.  Pairplot coloured by target  ------------------------------
# ──────────────────────────────────────────────────────────────
sns.pairplot(df,
             vars=X_num.columns[:5],   # sample first 5 num vars
             hue=target,
             diag_kind="kde")
plt.savefig(ADV_DIR/"pairplot.png")
plt.close()

# ──────────────────────────────────────────────────────────────
# 4.  UMAP (or t-SNE) 3-D embedding  ----------------------------
# ──────────────────────────────────────────────────────────────
try:
    import umap
    emb = umap.UMAP(n_neighbors=20, min_dist=.3, random_state=0)\
              .fit_transform(StandardScaler().fit_transform(X_num))
except ImportError:
    emb = TSNE(n_components=3, perplexity=30, random_state=0)\
        .fit_transform(StandardScaler().fit_transform(X_num))

emb_df = pd.DataFrame(emb, columns=["e1", "e2", "e3"])
emb_df[target] = df[target].values
sns.scatterplot(x="e1", y="e2", hue=target, data=emb_df, s=10)
plt.title("Low-dim embedding coloured by target")
plt.tight_layout()
plt.savefig(ADV_DIR/"embedding.png")
plt.close()

# ──────────────────────────────────────────────────────────────
# 5.  Leakage sniff: feature vs target timestamp  ---------------
# ──────────────────────────────────────────────────────────────
if {"last_login", target}.issubset(df.columns):
    future_rows = (df.last_login > df[target].index.map(
        lambda idx: df.last_login.iloc[idx])).mean()
    with open(ADV_DIR/"leakage_check.txt", "w") as fh:
        fh.write(f"future rows ratio: {future_rows:0.4f}\n")

# ──────────────────────────────────────────────────────────────
# 6.  Time-series Decomposition & ACF/PACF  ---------------------
# ──────────────────────────────────────────────────────────────
if "last_login" in df.columns:
    ts = (df.set_index("last_login")
            .resample("D")["amount"]
            .sum()
            .asfreq("D")
            .fillna(method="ffill"))
    decomp = sm.tsa.seasonal_decompose(ts, model="additive")
    decomp.plot()
    plt.tight_layout()
    plt.savefig(ADV_DIR/"ts_decompose.png")
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    plot_acf(ts, ax=ax[0])
    plot_pacf(ts, ax=ax[1])
    plt.tight_layout()
    plt.savefig(ADV_DIR/"acf_pacf.png")
    plt.close()

# ──────────────────────────────────────────────────────────────
# 7.  K-means elbow & silhouette  -------------------------------
# ──────────────────────────────────────────────────────────────
max_k = 10
ssd = []
sil = []
for k in range(2, max_k+1):
    km = KMeans(n_clusters=k, random_state=0).fit(X_num)
    ssd.append(km.inertia_)
    sil.append(pg.cluster_silhouette_score(X_num, km.labels_))

plt.plot(range(2, max_k+1), ssd, marker="o")
plt.xticks(range(2, max_k+1))
plt.ylabel("Sum-sq distance")
plt.xlabel("k")
plt.title("Elbow")
plt.tight_layout()
plt.savefig(ADV_DIR/"kmeans_elbow.png")
plt.close()

plt.plot(range(2, max_k+1), sil, marker="o")
plt.title("Silhouette")
plt.savefig(ADV_DIR/"kmeans_silhouette.png")
plt.close()

# ──────────────────────────────────────────────────────────────
# 8.  Auto-profilers  ------------------------------------------
# ─ ydata-profiling generates a full HTML (heavy)  -------------
if "ydata_profiling" in globals():
    prof = yp.ProfileReport(df, title="YData Profiling – EDA")
    prof.to_file(ADV_DIR/"profile.html")

# dabl quick-plot (lightweight)
try:
    import dabl
    dabl.plot(df, target_col=target)
    plt.savefig(ADV_DIR/"dabl_plot.png")
    plt.close()
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────
# 9.  Manifest logging  ----------------------------------------
manifest = {
    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    "rows": int(len(df)),
    "features": int(df.shape[1]),
    "profiling": "profile.html" if (ADV_DIR/"profile.html").exists() else "none"
}
(ADV_DIR/"manifest.json").write_text(json.dumps(manifest, indent=2))
print("✅ Advanced EDA done — artefacts in", ADV_DIR.relative_to(Path.cwd()))
