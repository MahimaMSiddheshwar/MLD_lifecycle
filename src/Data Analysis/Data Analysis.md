## 4 — Phase 4 · **Exploratory Data Analysis (EDA)**<a name="4-phase-4--exploratory-data-analysis"></a>

> **Goal** — get a _holistic view_ of the dataset, its distributions, relationships,
> and potential issues.
> This phase is orchestrated by **[`EDA.py`](src/Data%20Analysis/EDA.py)**, which
> reads the pre-processed data from `data/interim/clean.parquet` (output of Phase-3)
> and writes all artefacts to `reports/eda/`.
>
> - **[`EDA.py`](src/Data Analysis/EDA.py)** – univariate, bivariate, multivariate,
>   target-aware imbalance, leakage flags, optional HTML profile.
> - **[`EDA_advance.py`](src/Data Analysis/EDA_advance.py)** – still available for
>   very heavy add-ons (UMAP, t-SNE, time-series seasonality, etc.).

Both scripts read `data/interim/clean.parquet` (output of Phase-3) and write to
`reports/eda/`.

> downstream notebooks (or model cards) can embed.

---

### 4A Univariate Statistics & Plots<a name="4a-univariate-statistics--plots"></a>

| Metric / Test                                                         | Implementation                     | Output artefact                                         |
| --------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------- |
| mean, median, variance, std, skew, kurt                               | `df.amount.agg([...])`             | `reports/eda/univariate_summary.csv`                    |
| skew · kurt · IQR                                                     | `Series.skew()                     | kurt()`                                                 |
| Normality: Shapiro–Wilk, D’Agostino K², Jarque–Bera, Anderson–Darling | `scipy.stats`                      | CSV columns `shapiro_p`, `dagostino_p`, `jb_p`          |
| Normality p-values                                                    | Shapiro, D’Agostino, Jarque–Bera   | columns `shapiro_p`, `dagostino_p`, `jb_p`              |
| Visuals                                                               | Histogram + KDE, box-plot, QQ-plot | one PNG per numeric feature in `reports/eda/uva/plots/` |

> **Run only this section**
>
> ```bash
> python -m Data_Analysis.EDA --mode uva
> ```

---

### 4B Bivariate Tests & Visuals<a name="4b-bivariate-tests--visuals"></a>

| Pair Type        | Parametric                         | Non-Parametric        | Effect-size      |
| ---------------- | ---------------------------------- | --------------------- | ---------------- |
| num-num          | Pearson r                          | Spearman ρ, Kendall τ | `r²`,joint-plots |
| num vs 2 groups  | Welch-t                            | Mann–Whitney U        | Cohen’s d        |
| num vs k groups  | ANOVA                              | Kruskal–Wallis        | η²               |
| cat-cat          | χ²                                 | Fisher exact (2×2)    | Cramer V         |
| num ↔ num        | Pearson r · Spearman ρ · Kendall τ | optional              |
| num ↔ binary tgt | Point-Biserial r                   | effect-size in CSV    |
| num ↔ multi tgt  | Pearson r                          |

- **Joint-plot regressions** and **correlation heat-map** saved to  
  `reports/eda/bva/plots/`.
- Results table → `bivariate_summary.csv`.

Correlation heat-map & individual regressions are generated only when
`--pairplots` is passed.

```bash
python -m Data_Analysis.EDA --mode bva --target is_churn --pairplots
```

---

### 4C Multivariate Tests & Diagnostics<a name="4c-multivariate-tests--diagnostics"></a>

| Goal                   | Test / Tool                     | File / Visual                 |
| ---------------------- | ------------------------------- | ----------------------------- |
| Multi-collinearity     | max **VIF** across features     | `vif.csv`, `mva_summary.json` |
| Multivariate normality | **Mardia** P-val < 0.05         | `mva_summary.json`            |
| Overall association    | MANOVA (Pillai’s Trace)         | printed to console            |
| Dimensionality         | PCA scree ≥ 90 %                | `pca_scree.png`               |
| Cluster tendency       | **Hopkins** statistic           | `mva_summary.json`            |
| Heteroscedasticity     | **Breusch–Pagan** p-value       | `mva_summary.json`            |
| Correlation dendrogram | seaborn `clustermap`            | `corr_dendrogram.png`         |
| Leakage guard          | AUC ≈ 1 features → flagged JSON | `leakage_flags.json`          |

- **VIF**: Variance Inflation Factor, max VIF > 10 → multicollinearity
- **Mardia**: tests multivariate normality; p-value < 0.05 → reject H0
- **Hopkins**: tests cluster tendency; H0 = uniform distribution, H1 = clustering
- **Breusch–Pagan**: tests heteroscedasticity; p-value < 0.05 → reject H0
- **Dendrogram**: visualizes correlation structure; clusters of features
- **Leakage guard**: checks for future-timestamp overlap; flags features with AUC ≈ 1
- **PCA scree**: plots explained variance by components; helps decide dimensionality
- **MANOVA**: multivariate analysis of variance; checks if group means differ significantly
- **Pair-plots**: scatter matrix of numeric features, colored by target class

```bash
python -m Data_Analysis.EDA --mode mva --target is_churn
```

---

### 4D Advanced EDA — Mutual Info · Cramer-V · Embeddings · TS Decomp<a name="src/Data%20Analysis/EDA_advance.py"></a>

File: **[`EDA_advance.py`](src/Data%20Analysis/EDA_advance.py)**

What it adds on top of 4A-4C:

| Block                   | Highlight                                   |
| ----------------------- | ------------------------------------------- |
| Categorical association | **Cramer-V matrix** + mosaic plots          |
| Feature importance      | **Mutual Information** (numeric & one-hot)  |
| Interaction viz         | PairGrid by target, 2-D UMAP / 3-D t-SNE    |
| Leakage sniff           | Future-timestamp overlap check              |
| Time-series             | Seasonal decomposition, ACF/PACF plots      |
| Clustering quality      | k-means **elbow** + **silhouette** curves   |
| Auto-profilers          | `ydata_profiling` HTML, `dabl.plot` summary |

Outputs land in `reports/eda/advanced/`:

```bash
python -m Data_Analysis.EDA_advance
```

---

#### 🔍 Where to look after a run

```
reports/
└── eda/
    ├── univariate_summary.csv
    ├── bivariate_summary.csv
    ├── vif.csv
    ├── mva_summary.json
    ├── uva/plots/*.png
    ├── bva/plots/*.png
    ├── mva/plots/*.png
    └── advanced/
        ├── mutual_info.csv
        ├── profile.html
        └── *.png
```

---

### 🛠 CLI Cheat-Sheet

```bash
# lightweight (stats only)
python -m Data_Analysis.EDA --target is_churn

# full deep-dive with pair-plots + HTML profile
python -m Data_Analysis.EDA \
       --target is_churn \
       --pairplots \
       --profile
```

---

## 4·½. [Feature Selection & Early Train/Test Split](#4.5-phase-feature-select-split)

> **Why here?** Any statistic that _uses_ the target (variance filter,  
> mutual-information, Cramer-V, leakage sniff, etc.) must be learned on
> **training rows only**.  
> Therefore we:
>
> 1. **Split once — right now** (80 / 20 stratified by `target`  
>    or `--time-split` if temporal).
> 2. **Fit feature filters on _train_**, replay them on _val_ / _test_.
>    | Sub-step | Purpose | Script | Artefact |
>    | --------------------------- | ------------------------------------- | --------------------- | --------------------------------------------- |
>    | **4·½·0 Split** | Freeze leak-free `train / val / test` | `feature_selector.py` | `data/splits/*.parquet` `split_manifest.json` |
>    | **4·½·1 Low-variance drop** | remove near-constant cols | ″ | logged in manifest |
>    | **4·½·2 Target filter** | MI / chi² < threshold | ″ | `"kept","dropped"` lists |
>    | **4·½·3 Collinearity** | drop one of pairs with ρ > 0.95 | ″ | correlation heatmap |
>    | **4·½·4 Save plan** | Column lists for next phases | `"feature_plan.json"` |

```bash
 # full run – stratified split, MI filter @ 0.001, corr prune @ 0.95
 python -m Data_Analysis.feature_selector \
    --target is_churn \
    --mi-thresh 0.001 \
    --corr-thresh 0.95 \
    --seed 42
```

**Exit checklist** _ ✅ `data/splits/train.parquet` & `test.parquet` exist  
 _ ✅ `feature_plan.json` lists “keep” & “drop” columns  
 _ ✅ No feature on the **drop list** is referenced downstream  
 _ ✅ Issue **“Phase 4·½ Complete → start Phase 5 FE”** created

---
