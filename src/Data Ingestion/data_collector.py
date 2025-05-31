"""
omni_collector.py  — “one ring to rule them all” 🌐

A single class that **orchestrates every Phase-2 data-ingest channel**:
flat files, DBs (SQL + NoSQL), REST/GraphQL, streams, SaaS, IoT
…while adding privacy redaction, checksum logging, and lineage tags.

Dependencies (install only what you need):
  pip install pandas boto3 sqlalchemy psycopg2-binary pymongo requests
  pip install kafka-python gspread oauth2client paho-mqtt python-json-logger
  pip install great_expectations
"""
"""
omni_collector.py — Phase-2 “one-ring” data ingester 🌐
-------------------------------------------------------

* Flat files, S3 -> from_file
* SQL (Postgres / MySQL / BigQuery / Snowflake) -> from_sql
* NoSQL  (Mongo)    -> from_mongo
* REST / GraphQL    -> from_rest
* Kafka             -> from_kafka
* Google Sheets     -> from_gsheet
* MQTT IoT          -> from_mqtt
-------------------------------------------------------
Adds PII redaction, checksum logging, DVC-friendly snapshot saver,
and **optional Great Expectations validation**.
"""

from __future__ import annotations
import contextlib, hashlib, io, json, logging, os, re, time, pandas as pd
# ─── optional / lazy imports ────────────────────────────────────
with contextlib.suppress(ImportError):
    import boto3
    import requests
    import kafka
    import gspread
    from sqlalchemy import create_engine
    from pymongo import MongoClient
    import paho.mqtt.client as mqtt
with contextlib.suppress(ImportError):
    import great_expectations as ge                      # ← NEW

# ─── logging set-up ─────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOG_DIR / "ingest.log"),
              logging.StreamHandler()],
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("collector")

# ─── simple checksum helper ────────────────────────────────────


def _audit(df: pd.DataFrame, source: str) -> None:
    sha256 = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:12]
    log.info(f"{source:15} | rows={len(df):7,} | sha256={sha256}")


# ─── naïve PII masker (email & >10-digit phone) ────────────────
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\b\d{10,}\b")


def _mask(df: pd.DataFrame) -> pd.DataFrame:
    def scrub(x: str):                                # type: ignore
        if not isinstance(x, str):
            return x
        return PHONE.sub("[phone]", EMAIL.sub("[email]", x))
    return df.applymap(scrub)

# ╔══════════════════════════════════════════════════════════════╗
# ║                       OmniCollector                          ║
# ╚══════════════════════════════════════════════════════════════╝


class OmniCollector:
    """
    Universal data-ingest orchestrator.
    Parameters
    ----------
    pii_mask   : redact obvious PII (email / phone)        (default=True)
    validate   : run GE expectation suite after load       (default=False)
                 looks for `great_expectations/expectations/<suite>.yml`
    suite_name : which GE suite to run (defaults to          "sample_suite")
    """

    def __init__(self, *, pii_mask: bool = True,
                 validate: bool = False,
                 suite_name: str = "sample_suite") -> None:
        self.pii_mask = pii_mask
        self.validate = validate
        self.suite_name = suite_name

    # ── internal: validation helper ────────────────────────────
    def _validate(self, df: pd.DataFrame, source: str) -> None:
        if not self.validate:
            return
        if "great_expectations" not in globals():
            log.warning(
                "Great Expectations not installed — validation skipped")
            return
        ctx = ge.DataContext()                       # looks for ge/ directory
        suite = ctx.get_expectation_suite(f"{self.suite_name}")
        validator = ge.from_pandas(df)
        res = validator.validate(expectation_suite=suite)
        if not res.success:
            raise ValueError(f"🛑 GE validation failed for '{source}'")
        log.info(
            f"GE validation ✓  ({sum(m.success for m in res.results)}/{len(res.results)})")

    # ── 2A  Flat-file / Object-storage ─────────────────────────
    def from_file(self, path: str, **storage_opts) -> pd.DataFrame:
        if path.startswith("s3://"):
            import boto3
            bucket, key = path.split("s3://", 1)[1].split("/", 1)
            obj = boto3.client("s3").get_object(
                Bucket=bucket, Key=key, **storage_opts)
            buffer, suffix = io.BytesIO(obj["Body"].read()), Path(key).suffix
        else:
            buffer, suffix = path, Path(path).suffix

        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(buffer)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(buffer)
        else:
            df = pd.read_csv(buffer)

        if self.pii_mask:
            df = _mask(df)
        self._validate(df, "flat")
        _audit(df, f"flat:{Path(path).name}")
        return df

    # ── 2B  SQL ────────────────────────────────────────────────
    def from_sql(self, dsn: str, query: str) -> pd.DataFrame:
        df = pd.read_sql(query, create_engine(dsn))
        if self.pii_mask:
            df = _mask(df)
        self._validate(df, "sql")
        _audit(df, "sql")
        return df

    # ── 2C  Mongo ──────────────────────────────────────────────
    def from_mongo(self, uri: str, db: str, coll: str, query: dict | None = None):
        cur = MongoClient(uri)[db][coll].find(query or {})
        df = pd.DataFrame(list(cur)).drop(columns="_id", errors="ignore")
        if self.pii_mask:
            df = _mask(df)
        self._validate(df, "mongo")
        _audit(df, "mongo")
        return df

    # ── 2D  REST / GraphQL ─────────────────────────────────────
    def from_rest(self, url: str, *, params=None, headers=None):
        import requests
        payload = requests.get(
            url, params=params, headers=headers, timeout=20).json()
        df = pd.json_normalize(payload)
        if self.pii_mask:
            df = _mask(df)
        self._validate(df, "rest")
        _audit(df, "rest")
        return df

    # ── 2E  Kafka ──────────────────────────────────────────────
    def from_kafka(self, topic: str, *, bootstrap: str, batch=10, group_id="omni"):
        rows = []
        consumer = kafka.KafkaConsumer(
            topic, bootstrap_servers=bootstrap,
            group_id=group_id, auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda b: json.loads(b.decode())
        )
        for _ in range(batch):
            rows.append(next(consumer).value)
        df = pd.DataFrame(rows)
        if self.pii_mask:
            df = _mask(df)
        self._validate(df, "kafka")
        _audit(df, "kafka")
        return df

    # ── 2F  Google Sheets ──────────────────────────────────────
    def from_gsheet(self, *, sheet_key: str, creds_json: str):
        sheet = gspread.service_account(
            filename=creds_json).open_by_key(sheet_key).sheet1
        df = pd.DataFrame(sheet.get_all_records())
        if self.pii_mask:
            df = _mask(df)
        self._validate(df, "gsheet")
        _audit(df, "gsheet")
        return df

    # ── 2G  MQTT IoT ───────────────────────────────────────────
    def from_mqtt(self, *, broker: str, topic: str, timeout=5):
        rows = []
        def on_msg(client, _, msg): rows.append(json.loads(msg.payload))
        client = mqtt.Client()
        client.connect(broker)
        client.subscribe(topic)
        client.on_message = on_msg
        client.loop_start()
        time.sleep(timeout)
        client.loop_stop()
        df = pd.DataFrame(rows)
        if self.pii_mask:
            df = _mask(df)
        self._validate(df, "mqtt")
        _audit(df, "mqtt")
        return df

    # ── snapshot helper ───────────────────────────────────────
    def save(self, df: pd.DataFrame, name: str, fmt="parquet") -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fn = Path("data/raw") / f"{name}_{ts}.{fmt}"
        fn.parent.mkdir(parents=True, exist_ok=True)
        (df.to_parquet if fmt == "parquet" else df.to_csv)(fn, index=False)
        log.info(f"snapshot → {fn}")
        return fn


# =====================  EXAMPLE USAGE  =========================
# if __name__ == "__main__":
#    oc = OmniCollector()

#    csv_df   = oc.from_file("data/raw/users.csv")
#    s3_df    = oc.from_file("s3://my-bkt/2025/05/events.parquet")
#    pg_df    = oc.from_sql("postgresql://user:pwd@host/db",
        #  "SELECT * FROM public.events LIMIT 1000")
#    api_df   = oc.from_rest("https://api.exchangerate.host/latest")

#    kafka_df = oc.from_kafka("tx-events","localhost:9092",batch=50)
#    sheet_df = oc.from_gsheet(sheet_key=os.getenv("GS_KEY"),
        # creds_json="gcp-creds.json")

#    oc.save(csv_df,"users")
#    oc.save(api_df, "fxrates")
