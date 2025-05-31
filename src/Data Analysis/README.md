## 3 — Phase 3 · **Data Preparation**<a name="3-phase-3--data-preparation"></a>

> **Goal** — turn a raw snapshot from Phase-2 into a _model-ready_, versioned,
> privacy-hardened dataset in `data/processed/`, plus an interim copy in
> `data/interim/`.
> All logic lives in
> **[`src/ml_pipeline/prepare.py`](src/ml_pipeline/prepare.py)** —
> a configurable pipeline class (**`DataPreparer`**).

---

### 3A Schema Validation & Data Types<a name="3a-schema-validation--data-types"></a>

| Tool                        | What it does                                              | Where                                |
| --------------------------- | --------------------------------------------------------- | ------------------------------------ |
| **Pandera**                 | enforce column names, dtypes, value ranges, allowed enums | `schema = pa.DataFrameSchema({...})` |
| **pyjanitor**               | snake-cases column names (`df.clean_names()`)             | first line of `load_and_validate()`  |
| Data-quality tests (opt-in) | `great_expectations` (`--gx`)                             | `dq_validate()`                      |

**Why:** catch bad upstream changes early; guarantee downstream code never
breaks on dtype surprises.

---

### 3B.1 De-duplication & Invariant Pruning <a name="3b1-dedup"></a>

- `--dedup uid` → drops perfect-duplicate _rows_.
- `--prune-const 0.99` → removes columns where one value ≥ 99 %.

---

### 3B Missing-Value Strategy<a name="3b-missing-value-strategy"></a>

_Default_: median (numeric) + mode (categorical).
_Optional_: `--knn` flag enables **`KNNImputer`** (k=5).

| Technique      | Flag              | Notes                                |
| -------------- | ----------------- | ------------------------------------ |
| Median / Mode  | _(default)_       | fast & deterministic                 |
| **KNNImputer** | `--knn`           | non-linear numeric guess             |
| Drop column    | `--drop-miss 0.4` | removes any feature with > 40 % NaNs |
| Drop row       | `--drop-miss 0.4` | removes any row with > 40 % NaNs     |

```bash
python -m ml_pipeline.prepare --knn      # fancy impute
```

_Diagnostics:_ generates a `missingno.matrix` plot for the first 1 000 rows.

---

### 3C Outlier Detection & Treatment<a name="3c-outlier-detection--treatment"></a>

| Method             | Flag                      | Notes                          |                    |                            |
| ------------------ | ------------------------- | ------------------------------ | ------------------ | -------------------------- |
| IQR fence (1.5×)   | `--outlier iqr` (default) | quick & interpretable          |                    |                            |
| Z-score (          | z                         | < 3)                           | `--outlier zscore` | good for gaussian-ish data |
| Isolation Forest   | `--outlier iso`           | detects multivariate anomalies |                    |                            |
| Local Outlier Fac. | `--outlier lof`           | cluster-shaped data            |

---

### 3D Data Transformation & Scaling<a name="3d-data-transformation--scaling"></a>

| Transform                          | Flag                       | Comment                    |
| ---------------------------------- | -------------------------- | -------------------------- |
| log-transform on `amount`          | on by default (`np.log1p`) | stabilise heavy-tail       |
| **StandardScaler**                 | `--scaler standard`        | zero-mean / unit-var       |
| **RobustScaler** (IQR)             | `--scaler robust`          | heavy-outlier datasets     |
| **PowerTransformer (Yeo-Johnson)** | `--scaler yeo`             | make data closer to normal |

---

### 3E Class / Target Balancing<a name="3e-class-target-balancing"></a>

| Technique                   | Flag                 | Use-case                  |
| --------------------------- | -------------------- | ------------------------- |
| **SMOTE** over-sampling     | `--balance smote`    | minority boost            |
| **NearMiss** under-sampling | `--balance nearmiss` | huge majority down-sample |

```bash
python -m ml_pipeline.prepare --balance smote
```

---

### 3F Data Versioning & Lineage<a name="3f-data-versioning--lineage"></a>

- Saves **both** `data/interim/clean.parquet` (pre-scale) _and_
  `data/processed/scaled.parquet` (final).
- Writes `reports/lineage/prep_manifest.json`, e.g.

```jsonc
{
  "timestamp": "2025-05-30T12:42:01",
  "rows": 104876,
  "scaler": "robust",
  "outlier": "iso",
  "balance": "smote",
  "raw_sha": "7b12e0f83e01"
}
```

Add these files to **DVC** or **LakeFS** so every model build can
pin-point exactly which prep config & raw snapshot produced it.

---

### 3G Feature Pruning (High NaN / High Corr) <a name="3g-prune"></a>

- **NaN threshold** `--drop-miss p` → prune if NaNs > p
- **Corr threshold** `--drop-corr 0.95` → greedily drop highly-correlated pair

Manifest of drops saved to `reports/lineage/prune_log.json`.

---

### 🔧 Quick-Start Cheat-Sheet

```bash
# 1. Default happy-path (median/mode, IQR, standard scale)
python -m ml_pipeline.prepare

# 2. Robust pipeline for gnarly data
python -m ml_pipeline.prepare \
       --knn \
       --outlier iso \
       --scaler robust \
       --balance smote
```

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
