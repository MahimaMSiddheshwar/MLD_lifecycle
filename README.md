# 🤖 MLDLC – End-to-End Machine-Learning Project Template

This repository is a **ready-made workflow skeleton** for any ML project, small
or large.  
It captures the full _Machine-Learning Development Life-Cycle_ (MLDLC) in
clearly separated phases—so every new problem you tackle starts with the same,
battle-tested structure instead of an empty folder.

## Why use this repo?

- **Consistent anatomy** – one place for data, code, reports, and models
- **Phase gates** – checklists ensure you don’t jump ahead with fuzzy scope
- **Security & governance hooks** – PII masking, lineage, and basic compliance
- **Extensible** – each phase ships with a runnable Python stub you can swap or extend
- **Tool-agnostic** – works for tabular, NLP, vision, or time-series; local or cloud

## Repo Scaffold

```text
.
├── data/                         # raw/, interim/, processed/
├── src/
│   ├── Data Ingestion/
│   │   └── omni_collector.py     # Phase-2 engine  (⇨ see link below)
│   ├── Data Cleaning/
│   │   ├── data_preparation.py   # Prep pipeline  (Phase-3)
│   │   └── data_wrangling_cleaning.py
│   ├── Data Analysis/
│   │   ├── EDA.py                # Basic EDA      (Phase-4)
│   │   └── EDA_advance.py        # Advanced EDA   (Phase-4 extra)
│   └── Feature Engineering/
│       └── feature_engineering.py
├── notebooks/                    # Optional ad-hoc ipynb
├── reports/                      # Auto EDA, drift, model cards
├── models/                       # MLflow or on-disk artefacts
├── docker/                       # Dockerfile & helpers
├── dvc.yaml                      # DVC pipeline
├── pyproject.toml                # editable-install metadata
├── .github/                      # CI/CD workflows
└── README.md                     # ← you’re reading it
```

---

## Table of Contents

0. [Repo Scaffold](#0-repo-scaffold)

1. [Phase 1 — Problem Definition](#1-phase-1--problem-definition)

2. [Phase 2 — **Data Collection**](src/Data%20Ingestion/omni_collector.py)
   • [2A Flat-Files & Object Storage](#2a-flat-files--object-storage)
   • [2B Relational Databases](#2b-relational-databases)
   • [2C NoSQL & Analytical Stores](#2c-nosql--analytical-stores)
   • [2D APIs & Web Scraping](#2d-apis--web-scraping)
   • [2E Streaming & Message Queues](#2e-streaming--message-queues)
   • [2F SaaS & Cloud-Native Connectors](#2f-saas--cloud-native-connectors)
   • [2G Sensors & IoT](#2g-sensors--iot)
   • [2H Data Privacy & Governance Hooks](#2h-data-privacy--governance-hooks)
   • [2I Logging, Auditing & Checksums](#2i-logging-auditing--checksums)

3. [Phase 3 — **Data Preparation**](src/Data%20Cleaning/data_preparation.py)
   • [3A Schema Validation & Data Types](#3a-schema-validation--data-types)
   • [3B Missing-Value Strategy](#3b-missing-value-strategy)
   • [3C Outlier Detection & Treatment](#3c-outlier-detection--treatment)
   • [3D Data Transformation & Scaling](#3d-data-transformation--scaling)
   • [3E Class / Target Balancing](#3e-class-target-balancing)
   • [3F Data Versioning & Lineage](#3f-data-versioning--lineage)

4. [Phase 4 — **Exploratory Data Analysis (EDA)**](src/Data%20Analysis/EDA.py)
   • [4A Univariate Statistics & Plots](#4a-univariate-statistics--plots)
   • [4B Bivariate Tests & Visuals](#4b-bivariate-tests--visuals)
   • [4C Multivariate Tests & Diagnostics](#4c-multivariate-tests--diagnostics)
   • [4D Advanced EDA (MI · Cramer-V · Embeddings · TS Decomp)](src/Data%20Analysis/EDA_advance.py)

5. [Phase 5 — Feature Engineering](src/Feature%20Engineering/feature_engineering.py)
   • [5A Scaling & Normalization](#5a-scaling--normalization)
   • [5B Encoding Categorical Variables](#5b-encoding-categorical-variables)
   • [5C Handling Imbalanced Data](#5c-handling-imbalanced-data)
   • [5D Dimensionality Reduction](#5d-dimensionality-reduction)
   • [5E Automated Feature Synthesis](#5e-automated-feature-synthesis)
   • [5F Text / NLP Feature Extraction](#5f-text--nlp-feature-extraction)
   • [5G Image Feature Extraction](#5g-image-feature-extraction)
   • [5H Time-Series Feature Engineering](#5h-time-series-feature-engineering)

6. [Phase 6 — Model Design & Training](#6-phase-6--model-design--training)
   • [6A Algorithm Selection](#6a-algorithm-selection)
   • [6B Regularisation Techniques](#6b-regularisation-techniques)
   • [6C Cross-Validation Variants](#6c-cross-validation-variants)
   • [6D Hyper-Parameter Optimisation](#6d-hyper-parameter-optimisation)
   • [6E Early-Stopping & LR Scheduling](#6e-early-stopping--lr-scheduling)
   • [6F Ensembling & Bagging / Stacking](#6f-ensembling--bagging--stacking)
   • [6G Data Augmentation & Noise Injection](#6g-data-augmentation)

7. [Phase 7 — **Evaluation, Regularisation Audit & Hardening**](#7-phase-7--evaluation-regularisation--hardening)
   • [7A Core Metrics](#7a-core-metrics)
   • [7B Calibration & Probabilistic Quality](#7b-calibration--probability-quality)
   • [7C Bias / Fairness & Group Metrics](#7c-bias--fairness)
   • [7D Explainability (SHAP · LIME · XAI)](#7d-explainability)
   • [7E Robustness & Adversarial Testing](#7e-robustness--adversarial-testing)
   • [7F Over-fitting Diagnostics](#7f-over-fitting-diagnostics)
   • [7G Model Card & Governance Sign-off](#7g-model-card--governance)

8. [Phase 8 — **Deployment & Serving**](#8-phase-8--deployment--serving)
   • [8A Model Serialization](#8a-model-serialization)
   • [8B Packaging & Containerization](#8b-packaging--containerization)
   • [8C API & Micro-service Layer](#8c-api--micro-service-layer)
   • [8D Inference Optimisation](#8d-inference-optimisation)
   • [8E CI/CD & Model-Registry Promotion](#8e-cicd--model-registry-promotion)
   • [8F Release Strategies](#8f-release-strategies)
   • [8G Runtime Security](#8g-runtime-security)

9. [Phase 9 — **Monitoring, Drift & Retraining**](#9-phase-9--monitoring-drift--retraining)
   • [9A Performance & Latency Metrics](#9a-performance--latency-metrics)
   • [9B Data & Concept Drift Detection](#9b-data--concept-drift-detection)
   • [9C Model Quality Tracking & Alerts](#9c-model-quality-tracking--alerts)
   • [9D Logging & Audit Trails](#9d-logging--audit-trails)
   • [9E Automated Retraining Pipelines](#9e-automated-retraining-pipelines)
   • [9F Rollback / Roll-forward Playbooks](#9f-rollback--roll-forward-playbooks)
   • [9G Continuous Compliance & Model Registry](#9g-continuous-compliance--model-registry)

10. [Cloud-Security Pillars](#10-cloud-security-pillars)

11. [CI/CD & Automation](#11-cicd--automation)

12. [FAQ](#12-faq)

13. [License](#13-license)

## 1 — Phase 1 · Problem Definition<a name="1-phase-1--problem-definition"></a>

> **Goal** — turn a fuzzy idea into an implementable, testable ML plan..
> A condensed checklist I’ve evolved after dozens of production projects

---

### 1 Clarify the Business “Why”

| Ask                                                 | Why it matters                                         |
| --------------------------------------------------- | ------------------------------------------------------ |
| _“If the model is perfect tomorrow, what changes?”_ | forces ROI thinking, surfaces hidden KPIs              |
| _“Who loses sleep if this fails?”_                  | reveals actual decision-makers / veto-holders          |
| _“What is the cost of a wrong prediction?”_         | calibrates class-imbalance weighting, thresholds, SLAs |

---

### 2 Translate to an ML Task

1. **Prediction vs. ranking vs. clustering?**  
   Map to _supervised_, _recommender_, _unsupervised_ or _forecasting_ bucket.
2. **Unit of prediction** (row-level? session? account? pixel?).  
   Mis-scoping here kills performance later.
3. **Latency tolerance** → batch, near-real-time, or streaming.

> _Tip_: if you can’t phrase the target as a column in a future CSV,  
> you don’t have a learnable task yet.

---

### 3 Do a Data Reality Check _before_ Deep EDA

- Column availability at **prediction time** (no future leakage).
- Volume vs. freshness vs. drift risk.
- Quick uni-variate histograms → smell test for PII, bogus zeros, unit errors.
- **Baselines**: random, constant, or simple ruleset.  
  If a baseline already beats the target KPI, challenge the need for ML.

---

### 4 Sketch the End-to-End Flow on One Whiteboard

```mermaid
flowchart LR
    subgraph Offline
        A[Raw Sources] --> B[Data Prep<br>+ EDA] --> C[Train / Tune]
    end
    subgraph Online
        C --> D(Model Artifact) --> E[API / Batch Job]
    end
    E -->|logs| F[Monitoring & Drift]
    F -->|trigger| B
```

### 5 Problem-Clarity **Exit Checklist**

_(all boxes must be ticked before Phase 2 – Data Collection – may start)_

| ⬜  | Requirement                                               | Hint                                                    |
| --- | --------------------------------------------------------- | ------------------------------------------------------- |
|     | **Business objective** phrased as one SMART sentence      | “Reduce voluntary churn by 15 % within 2 quarters”      |
|     | **Unit of analysis** defined                              | “prediction per _customer-ID_ per month”                |
|     | **Target variable** unambiguously stated and time-stamped | `is_churn` ∈ {0, 1} measured 30 days after billing date |
|     | **Primary success metric** and numeric threshold agreed   | “F1 ≥ 0.82 on 2024-Q4 hold-out”                         |
|     | **Constraints & assumptions** captured                    | latency, region, budget, feature freeze date            |
|     | **High-level ethical / bias risks** listed                | sensitive attributes, potential exclusion harms         |
|     | **Regulatory touch-points** identified                    | GDPR/CCPA, sector-specific rules                        |
|     | **Baseline approach** written down                        | random or simple heuristic score                        |
|     | All above items reviewed & signed off (email / doc)       | attach link in project tracker                          |

> When the table is fully checked, create an issue titled  
> **“Phase-1 Complete – proceed to Data Collection”** and assign it to the team lead.  
> Only then move on to Phase-2.

## 2 — Phase 2 · Data Collection<a name="2-phase-2--data-collection"></a>

> **Goal** — pull data from _any_ source, stamp it with lineage, mask PII, and
> persist an immutable snapshot in `data/raw/` that DVC (or LakeFS) can track.
> The heavy lifting is baked into **[`OmniCollector`](src/data_ingest/data_collector.py)**;
> the subsections below show how each channel maps to one collector method,
> plus security/gov-hooks you should enable in production.

---

### 2A Flat-Files & Object Storage<a name="2a-flat-files--object-storage"></a>

| Format                    | Example call                                         | Notes                                                       |
| ------------------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| **CSV / TSV**             | `oc.from_file("data/raw/users.csv")`                 | Auto-detects delimiter.                                     |
| **Excel**                 | `oc.from_file("marketing.xlsx")`                     | Supports multiple sheets (`pd.read_excel(sheet_name=...)`). |
| **Parquet / ORC / Avro**  | `oc.from_file("events.parquet")`                     | Requires `pyarrow`.                                         |
| **S3 / GCS / Azure Blob** | `oc.from_file("s3://my-bkt/2025/05/events.parquet")` | Pass `storage_options` → KMS, STS, IAM role.                |
| **ZIP / TAR**             | `oc.from_file("archive.zip")`                        | Auto-extracts first file if single-member.                  |

_Governance_: set bucket-policy to SSE-KMS, use **least-privilege IAM**; the
collector runs regex-based email/phone redaction before snapshot-save.

---

### 2B Relational Databases<a name="2b-relational-databases"></a>

```python
dsn   = "postgresql+psycopg2://ml_user:${PG_PWD}@pg-ro.acme.local:5432/warehouse"
query = "SELECT uid, age, churn_flag, ts FROM analytics.users WHERE ts >= NOW()-INTERVAL '90 days'"
df    = oc.from_sql(dsn, query)
```

_Extras_:

- parameterised queries to avoid SQLi
- use **read-only replica endpoints**
- column-level encryption with pgcrypto (Postgres) or TDE (MySQL 8+)

---

### 2C NoSQL & Analytical Stores<a name="2c-nosql--analytical-stores"></a>

```python
df = oc.from_mongo("mongodb://ro_user:${MONGO_PWD}@mongo-ro:27017",
                   db="crm", coll="events",
                   query={"ts": {"$gte": "2025-01-01"}})
```

BigQuery & Snowflake are available via `oc.from_sql(...)`
because they expose JDBC/SQLAlchemy drivers.

---

### 2D APIs & Web Scraping<a name="2d-apis--web-scraping"></a>

```python
df_fx = oc.from_rest("https://api.exchangerate.host/latest",
                     params={"base": "USD"})
```

If you must scrape:

```python
from bs4 import BeautifulSoup, requests
html = requests.get("https://example.com/pricelist", timeout=15).text
price_df = pd.read_html(str(BeautifulSoup(html,"lxml").find("table")))[0]
oc.save(price_df, "price_table")
```

_Security_: respect robots.txt, user-agent throttling, rotate tokens.

---

### 2E Streaming / Message Queues<a name="2e-streaming--message-queues"></a>

```python
# Consume the last 100 Kafka messages (JSON) without committing offsets
stream_df = oc.from_kafka(topic="tx-events",
                          bootstrap="kafka-broker:9092",
                          batch=100, group_id="omni-probe")
```

_Checkpointing_: commit offsets only after `oc.save()` succeeds,
so failed runs can re-process safely.

---

### 2F SaaS & Cloud-Native Connectors<a name="2f-saas--cloud-native-connectors"></a>

```python
df_sheet = oc.from_gsheet(sheet_key=os.getenv("GSHEET_ID"),
                          creds_json="gcp-sa.json")
```

Need HubSpot, Stripe, Salesforce?
Either:

1. Call their REST/Bulk API → `oc.from_rest()`, or
2. Use Fivetran / Airbyte to land data in Postgres/Snowflake, then `from_sql`.

---

### 2G Sensors & IoT Ingestion<a name="2g-sensors--iot"></a>

```python
iot_df = oc.from_mqtt(broker="192.168.1.50",
                      topic="factory/line1/#",
                      timeout=10)         # seconds to listen
```

Store raw telemetry uncompressed → `Parquet+ZSTD` later in an Apache Iceberg or
TimescaleDB bucket for long-term analytics.

---

### 2H Data Privacy & Governance Hooks<a name="2h-data-privacy--governance-hooks"></a>

- Built-in regex scrub for **emails** and **phone numbers**
- Extend `_mask()` to hash SSNs, tokenize names (use Bloom filter / format-preserving encryption).
- Tag snapshots with `dataset`, `source_system`, and `sensitivity` in DVC
  (`dvc params`) for future lineage queries.

---

### 2I Logging, Auditing & Checksums<a name="2i-logging-auditing--checksums"></a>

Every collector call:

1. **SHA-256** of the CSV bytes (or canonical Parquet bytes)
2. **row-count**
3. **source label**
4. UTC timestamp

is appended to `logs/ingest.log`, e.g.

```
2025-05-30T23:14:09 | INFO | flat:events.parquet  | rows= 104 876 | sha256=7b12e0f83e01
```

Use this file plus DVC commit history for a tamper-evident audit trail.

---

### 🔧 Quick-Start Recap

```bash
# install in editable mode
pip install -e .

# CLI one-liner pulls CSV and snapshots into data/raw/
omni-collect file data/raw/users.csv

# REST example
omni-collect rest https://api.exchangerate.host/latest
```

`omni-collect` is defined in `pyproject.toml` under `[project.scripts]`
and implemented in **`src/data_ingest/omni_cli.py`**, which wraps the
same `OmniCollector` methods shown above.

## 3 — Phase 3 · **Data Preparation**<a name="3-phase-3--data-preparation"></a>

> **Goal** — turn a raw snapshot from Phase-2 into a _model-ready_, versioned,
> privacy-hardened dataset in `data/processed/`, plus an interim copy in
> `data/interim/`.
> All logic lives in
> **[`src/ml_pipeline/prepare.py`](src/ml_pipeline/prepare.py)** —
> a configurable pipeline class (**`DataPreparer`**).

---

### 3A Schema Validation & Data Types<a name="3a-schema-validation--data-types"></a>

| Tool          | What it does                                              | Where                                |
| ------------- | --------------------------------------------------------- | ------------------------------------ |
| **Pandera**   | enforce column names, dtypes, value ranges, allowed enums | `schema = pa.DataFrameSchema({...})` |
| **pyjanitor** | snake-cases column names (`df.clean_names()`)             | first line of `load_and_validate()`  |

**Why:** catch bad upstream changes early; guarantee downstream code never
breaks on dtype surprises.

---

### 3B Missing-Value Strategy<a name="3b-missing-value-strategy"></a>

_Default_: median (numeric) + mode (categorical).
_Optional_: `--knn` flag enables **`KNNImputer`** (k=5).

```bash
python -m ml_pipeline.prepare --knn      # fancy impute
```

_Diagnostics:_ generates a `missingno.matrix` plot for the first 1 000 rows.

---

### 3C Outlier Detection & Treatment<a name="3c-outlier-detection--treatment"></a>

| Method           | Flag                      | Notes                          |                    |                            |
| ---------------- | ------------------------- | ------------------------------ | ------------------ | -------------------------- |
| IQR fence (1.5×) | `--outlier iqr` (default) | quick & interpretable          |                    |                            |
| Z-score (        | z                         | < 3)                           | `--outlier zscore` | good for gaussian-ish data |
| Isolation Forest | `--outlier iso`           | detects multivariate anomalies |                    |                            |

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

> Two Python scripts power this phase:
>
> - **[`EDA.py`](src/Data%20Analysis/EDA.py)** – core stats & plots (4A-4C)
> - **[`EDA_advance.py`](src/Data%20Analysis/EDA_advance.py)** – deep-dive add-ons (4D)
>
> Both read the interim parquet produced by Phase-3 (`data/interim/clean.parquet`)  
> and populate `reports/eda/…` with CSV summaries, PNGs, and manifests that
> downstream notebooks (or model cards) can embed.

---

### 4A Univariate Statistics & Plots<a name="4a-univariate-statistics--plots"></a>

| Metric / Test                                                         | Implementation                     | Output artefact                                         |
| --------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------- |
| mean, median, variance, std, skew, kurt                               | `df.amount.agg([...])`             | `reports/eda/univariate_summary.csv`                    |
| IQR                                                                   | `q3 - q1`                          | same CSV                                                |
| Normality: Shapiro–Wilk, D’Agostino K², Jarque–Bera, Anderson–Darling | `scipy.stats`                      | CSV columns `shapiro_p`, `dagostino_p`, `jb_p`          |
| Outliers flag                                                         | Z-score / IQR fence, density plot  | embedded inside PNGs                                    |
| Visuals                                                               | Histogram + KDE, box-plot, QQ-plot | one PNG per numeric feature in `reports/eda/uva/plots/` |

> **Run only this section**
>
> ```bash
> python -m Data_Analysis.EDA --mode uva
> ```

---

### 4B Bivariate Tests & Visuals<a name="4b-bivariate-tests--visuals"></a>

| Pair Type       | Parametric | Non-Parametric        | Effect-size |
| --------------- | ---------- | --------------------- | ----------- |
| num-num         | Pearson r  | Spearman ρ, Kendall τ | `r²`        |
| num vs 2 groups | Welch-t    | Mann–Whitney U        | Cohen’s d   |
| num vs k groups | ANOVA      | Kruskal–Wallis        | η²          |
| cat-cat         | χ²         | Fisher exact (2×2)    | Cramer V    |

- **Joint-plot regressions** and **correlation heat-map** saved to  
  `reports/eda/bva/plots/`.
- Results table → `bivariate_summary.csv`.

```bash
python -m Data_Analysis.EDA --mode bva
```

---

### 4C Multivariate Tests & Diagnostics<a name="4c-multivariate-tests--diagnostics"></a>

| Goal                   | Test / Tool             | File / Visual         |
| ---------------------- | ----------------------- | --------------------- |
| Multicollinearity      | VIF                     | `vif.csv`             |
| Multivariate normality | **Mardia** P-value      | `mva_summary.json`    |
| Overall association    | MANOVA (Pillai’s Trace) | printed to console    |
| Dimensionality         | PCA scree ≥ 90 %        | `pca_scree.png`       |
| Cluster tendency       | Hopkins statistic       | manifest              |
| Heteroscedasticity     | Breusch–Pagan           | manifest              |
| Correlation dendrogram | seaborn `clustermap`    | `corr_dendrogram.png` |

```bash
python -m Data_Analysis.EDA --mode mva
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
