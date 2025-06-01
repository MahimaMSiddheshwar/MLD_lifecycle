## 2 — Phase 2 · **Data Collection & Initial Validation**<a name="2-phase-2--data-collection"></a>

> **Goal** — pull tabular data from _any_ source, redact obvious PII,  
> version the raw snapshot in `data/raw/`, **and fail-fast** if the feed violates
> basic quality expectations.  
> Everything is orchestrated by **[`OmniCollector`](src/Data%20Ingestion/data_collector.py)**  
> (the “one ring” in Phase-2).  
> When the collector finishes it hands a Parquet file to Phase-3 and writes an
> audit-trail line to `logs/ingest.log`.

### 2·0 What happens under the hood 🛠

1. **Download / query / consume** via one of `OmniCollector.from_*` channels
2. **Regex PII scrub** (email & phone) → _Data Privacy Hook_
3. **Great Expectations suite** runs (schema, non-null %, range checks, etc.)
   - Fails the pipeline if any _critical_ expectation is violated
4. **SHA-256 & row-count** logged to `logs/ingest.log`
5. Snapshot saved to `data/raw/…` and git-/DVC-tracked

> Copy `great_expectations/expectations/sample_suite.yml` and tailor it to your
> dataset; the default checks only shape & non-null counts.

---

### 2A Flat-Files & Object Storage<a name="2a-flat-files--object-storage"></a>

| Format / Location         | Collector call                                    | Notes                                       |
| ------------------------- | ------------------------------------------------- | ------------------------------------------- |
| **CSV / TSV**             | `oc.from_file("data/raw/users.csv")`              | Auto-delimiter / Pandas dtype inference     |
| **Excel**                 | `oc.from_file("marketing.xlsx")`                  | Multi-sheet supported                       |
| **Parquet / ORC / Avro**  | `oc.from_file("events.parquet")`                  | Needs **pyarrow**                           |
| **S3 / GCS / Azure Blob** | `oc.from_file("s3://bucket/path/events.parquet")` | IAM role / KMS handled by `storage_options` |
| **ZIP / TAR**             | `oc.from_file("archive.zip")`                     | Auto-extract if single-member archive       |

_Governance_: set bucket-policy to SSE-KMS, use **least-privilege IAM**; the
collector runs regex-based email/phone redaction before snapshot-save.

---

### 2B Relational DBs<a name="2b-relational-databases"></a>

```python
dsn   = "postgresql+psycopg2://ml_user:${PG_PWD}@pg-ro.acme.local/warehouse"
query = """
SELECT uid, age, is_churn, updated_at
FROM   analytics.users
WHERE  updated_at >= CURRENT_DATE - INTERVAL '90 days'
"""
df = oc.from_sql(dsn, query)
```

_Parameterised queries_ avoid SQL-i, and read-only replicas protect prod.

---

### 2C NoSQL / Analytical Stores<a name="2c-nosql--analytical-stores"></a>

```python
df = oc.from_mongo(
        uri="mongodb://ro_user:${MONGO_PWD}@mongo-ro:27017",
        db="crm", coll="events",
        query={"updated_at": {"$gte": "2025-01-01"}}
)
```

(BigQuery & Snowflake use `from_sql` via SQLAlchemy drivers.)

---

### 2D APIs & Web Scraping<a name="2d-apis--web-scraping"></a>

```python
df_fx = oc.from_rest(
          "https://api.exchangerate.host/latest",
          params={"base": "USD"}
)
```

Scraping? Use `BeautifulSoup` or Playwright, then `oc.save(df, "source_name")`.

If you must scrape:

```python
from bs4 import BeautifulSoup, requests
html = requests.get("https://example.com/pricelist", timeout=15).text
price_df = pd.read_html(str(BeautifulSoup(html,"lxml").find("table")))[0]
oc.save(price_df, "price_table")
```

_Security_: respect robots.txt, user-agent throttling, rotate tokens.

### 2E Streams / Message Queues<a name="2e-streaming--message-queues"></a>

```python
stream_df = oc.from_kafka(
               topic="tx-events",
               bootstrap="broker:9092",
               batch=1_000,
               group_id="ingest-probe"
)
```

Offsets committed **after** `oc.save()` ➜ at-least-once semantics.

---

### 2F SaaS / Cloud-Native<a name="2f-saas--cloud-native-connectors"></a>

```python
df_sheet = oc.from_gsheet(
              sheet_key=os.getenv("GSHEET_ID"),
              creds_json="gcp-sa.json"
)
```

Need HubSpot, Stripe, Salesforce?
Either:

1. Call their REST/Bulk API → `oc.from_rest()`, or
2. Use Fivetran / Airbyte to land data in Postgres/Snowflake, then `from_sql`.

---

### 2G Sensors & IoT<a name="2g-sensors--iot"></a>

```python
iot_df = oc.from_mqtt(
            broker="192.168.1.50",
            topic="factory/line1/#",
            timeout=10
)
```

Store raw telemetry uncompressed → `Parquet+ZSTD` later in an Apache Iceberg or
TimescaleDB bucket for long-term analytics.

---

### 2H Data-Privacy & Governance Hooks<a name="2h-data-privacy--governance-hooks"></a>

- Regex redaction for emails & phone numbers (`_mask`)
- Extendable: plug your own `re` patterns or FPE tokenisers
- Lineage tags stored alongside the Parquet metadata (`dataset`, `source_system`, `run_id`)

---

### 2I Logging, Auditing & Checksums<a name="2i-logging-auditing--checksums"></a>

Every collector call:

1. **SHA-256** of the CSV bytes (or canonical Parquet bytes)
2. **row-count**
3. **source label**
4. UTC timestamp

Each ingest line in `logs/ingest.log`:

```
2025-05-30T22:14:09Z | INFO | flat:events.parquet | rows=104 876 | sha256=7b12e0f83e01
```

This + the DVC commit = tamper-evident audit trail.

---

### 🔧 Quick-Start

```bash
# ① install core + sqlalchemy + great_expectations
pip install -e .[ingest,sql,ge]

# ② pull CSV snapshot
omni-collect file data/raw/users.csv

# ③ check the log & Great Expectations report (html in great_expectations/…)
```

---

### 📌 Why the extra validation step?

1. **Fail-fast** – bad schemas blow up here, not during model training
2. **Confidence for 50 k engineers** – everyone inherits a baseline of QA
3. **CI-friendly** – the GE suite runs in GitHub Actions so broken feeds block the PR

If you need stricter checks (e.g. “< 5 % nulls in `age`”), edit
`great_expectations/expectations/sample_suite.yml`.

---

> **Next phase ➜ [Data Preparation](#3-phase-3--data-preparation)**
> You now have an immutable, validated snapshot ready for cleaning & scaling.

---
