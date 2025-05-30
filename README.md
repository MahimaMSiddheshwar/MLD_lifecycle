````markdown
## 0 — Repo Scaffold<a name="0-repo-scaffold"></a>

```text
.
├── data/            # raw/, interim/, processed/ partitions
├── src/             # Python packages (pip-installable)
├── notebooks/       # Exploratory Jupyter work
├── reports/         # Auto-generated EDA, drift, model cards
├── models/          # MLflow or on-disk model artefacts
├── docker/          # Dockerfile & container helpers
├── dvc.yaml         # Data-Version-Control pipeline
├── .github/         # CI/CD workflows
└── README.md        # ← this file
````

---

## 🗂️ Table of Contents — *granular & exhaustive*

0. [Repo Scaffold](#0-repo-scaffold)

1. [Phase 1 — Problem Definition](#1-phase-1--problem-definition)

2. [Phase 2 — **Data Collection**](src/data_ingest/omni_collector.py)
   • [2A Flat-Files & Object Storage](#2a-flat-files--object-storage)
   • [2B Relational Databases](#2b-relational-databases)
   • [2C NoSQL & Analytical Stores](#2c-nosql--analytical-stores)
   • [2D APIs & Web Scraping](#2d-apis--web-scraping)
   • [2E Streaming & Message Queues](#2e-streaming--message-queues)
   • [2F SaaS & Cloud-Native Connectors](#2f-saas--cloud-native-connectors)
   • [2G Sensors & IoT](#2g-sensors--iot)
   • [2H Data Privacy & Governance Hooks](#2h-data-privacy--governance-hooks)
   • [2I Logging, Auditing & Checksums](#2i-logging-auditing--checksums)

3. [Phase 3 — **Data Preparation**](#3-phase-3--data-preparation)
   • [3A Schema Validation & Data Types](#3a-schema-validation--data-types)
   • [3B Missing-Value Strategy](#3b-missing-value-strategy)
   • [3C Outlier Detection & Treatment](#3c-outlier-detection--treatment)
   • [3D Data Transformation & Scaling](#3d-data-transformation--scaling)
   • [3E Class/Target Balancing](#3e-classtarget-balancing)
   • [3F Data Versioning & Lineage](#3f-data-versioning--lineage)

4. [Phase 4 — **Exploratory Data Analysis (EDA)**](#4-phase-4--exploratory-data-analysis)
   • [4A Univariate Statistics & Plots](#4a-univariate-statistics--plots)
   • [4B Bivariate Tests & Visuals](#4b-bivariate-tests--visuals)
   • [4C Multivariate Tests & Diagnostics](#4c-multivariate-tests--diagnostics)

5. [Phase 5 — Feature Engineering](#5-phase-5--feature-engineering)
   • [5A Scaling & Normalization](#5a-scaling--normalization)
   • [5B Encoding Categorical Variables](#5b-encoding-categorical-variables)
   • [5C Handling Imbalanced Data](#5c-handling-imbalanced-data)
   • [5D Dimensionality Reduction](#5d-dimensionality-reduction)
   • [5E Automated Feature Synthesis](#5e-automated-feature-synthesis)
   • [5F Text / NLP Feature Extraction](#5f-text--nlp-feature-extraction)
   • [5G Image Feature Extraction](#5g-image-feature-extraction)
   • [5H Time-Series Feature Engineering](#5h-time-series-feature-engineering)

6. [Phase 6 — Model Design & Training](#6-phase-6--model-design--training)

7. [Phase 7 — Evaluation, Regularisation & Hardening](#7-phase-7--evaluation-regularisation--hardening)

8. [Phase 8 — Deployment & Serving](#8-phase-8--deployment--serving)

9. [Phase 9 — Monitoring, Drift & Retraining](#9-phase-9--monitoring-drift--retraining)

10. [Cloud-Security Pillars](#10-cloud-security-pillars)

11. [CI/CD & Automation](#11-cicd--automation)

12. [FAQ](#12-faq)

13. [License](#13-license)

```
```
