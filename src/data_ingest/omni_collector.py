"""
omni_collector.py  — “one ring to rule them all” 🌐

A single class that **orchestrates every Phase-2 data-ingest channel**:
flat files, DBs (SQL + NoSQL), REST/GraphQL, streams, SaaS, IoT
…while adding privacy redaction, checksum logging, and lineage tags.

Dependencies (install only what you need):
pip install pandas boto3 sqlalchemy psycopg2-binary pymongo requests
pip install kafka-python gspread oauth2client paho-mqtt python-json-logger
"""

from __future__ import annotations
import os, io, json, hashlib, logging, time, gzip, base64, contextlib
from pathlib import Path
from datetime import datetime
import pandas as pd

# ---- optional libs (import lazily) ---------------------------
with contextlib.suppress(ImportError):
    import boto3
    from sqlalchemy import create_engine
    from pymongo import MongoClient
    import requests, gspread, kafka, paho.mqtt.client as mqtt

# ---- logging -------------------------------------------------
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOG_DIR / "ingest.log"),
              logging.StreamHandler()],
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("collector")

# ---- helper: checksum & audit row-count ----------------------
def _audit(df: pd.DataFrame, source: str) -> None:
    raw_bytes = df.to_csv(index=False).encode()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()[:12]
    log.info(f"{source:15} | rows={len(df):7,} | sha256={sha256}")

# ---- helper: basic PII masker (email & phone regex) ----------
import re
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE = re.compile(r"\b\d{10,}\b")
def _mask(df: pd.DataFrame) -> pd.DataFrame:
    def scrub(x: str) -> str:
        if not isinstance(x,str): return x
        x = EMAIL.sub("[email_redacted]", x)
        return PHONE.sub("[phone_redacted]", x)
    return df.applymap(scrub)

# ==============================================================
class OmniCollector:
    """Universal Phase-2 data collector with governance hooks."""

    def __init__(self, pii_mask: bool = True):
        self.pii_mask = pii_mask

    # ---------- 2A  Flat files / Object storage ----------------
    def from_file(self, path: str, **storage_opts) -> pd.DataFrame:
        """CSV / Parquet / Excel / (S3 URI if boto3 configured)."""
        if path.startswith("s3://"):
            s3 = boto3.client("s3")
            bucket, key = path.split("s3://")[1].split("/",1)
            obj = s3.get_object(Bucket=bucket, Key=key, **storage_opts)
            data = io.BytesIO(obj["Body"].read())
            suffix = Path(key).suffix
        else:
            data  = path
            suffix= Path(path).suffix

        if suffix in {".parquet",".pq"}:
            df = pd.read_parquet(data)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(data)
        else:                                  # default CSV/TSV
            df = pd.read_csv(data)

        if self.pii_mask: df = _mask(df)
        _audit(df, f"flat:{Path(path).name}")
        return df

    # ---------- 2B  Relational DBs ------------------------------
    def from_sql(self, dsn: str, query: str) -> pd.DataFrame:
        engine = create_engine(dsn)
        df = pd.read_sql(query, engine)
        if self.pii_mask: df = _mask(df)
        _audit(df, "sql")
        return df

    # ---------- 2C  NoSQL / analytical --------------------------
    def from_mongo(self, uri: str, db: str, coll: str, query: dict=None):
        query = query or {}
        client = MongoClient(uri)
        cur = client[db][coll].find(query)
        df = pd.DataFrame(list(cur))
        df.drop(columns="_id", errors="ignore", inplace=True)
        if self.pii_mask: df = _mask(df)
        _audit(df, "mongo")
        return df

    # ---------- 2D  REST / GraphQL / Scraping ------------------
    def from_rest(self, url: str, params: dict=None, headers: dict=None):
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        payload = r.json()
        df = pd.json_normalize(payload)
        if self.pii_mask: df = _mask(df)
        _audit(df, "rest")
        return df

    # ---------- 2E  Streaming (Kafka example) -------------------
    def from_kafka(self, topic: str, bootstrap: str, batch=10, group_id="omni"):
        consumer = kafka.KafkaConsumer(topic,
                                       bootstrap_servers=bootstrap,
                                       group_id=group_id,
                                       auto_offset_reset="latest",
                                       enable_auto_commit=False,
                                       value_deserializer=lambda x: json.loads(x.decode()))
        rows=[]
        for _ in range(batch):
            msg = next(consumer)
            rows.append(msg.value)
        df = pd.DataFrame(rows)
        if self.pii_mask: df = _mask(df)
        _audit(df, "kafka")
        return df

    # ---------- 2F  SaaS  (Google Sheets example) ---------------
    def from_gsheet(self, sheet_key: str, creds_json: str):
        gc = gspread.service_account(filename=creds_json)
        sheet = gc.open_by_key(sheet_key).sheet1
        df = pd.DataFrame(sheet.get_all_records())
        if self.pii_mask: df = _mask(df)
        _audit(df, "gsheet")
        return df

    # ---------- 2G  IoT / MQTT ---------------------------------
    def from_mqtt(self, broker:str, topic:str, timeout=5):
        rows=[]
        def on_message(client, _, msg):
            rows.append(json.loads(msg.payload))
        client = mqtt.Client()
        client.connect(broker); client.subscribe(topic)
        client.on_message = on_message
        client.loop_start(); time.sleep(timeout); client.loop_stop()
        df = pd.DataFrame(rows)
        if self.pii_mask: df = _mask(df)
        _audit(df, "mqtt")
        return df

    # ---------- utility to write to DVC-tracked raw layer -------
    def save(self, df: pd.DataFrame, name: str, fmt: str="parquet") -> Path:
        ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fn  = Path("data/raw")/f"{name}_{ts}.{fmt}"
        fn.parent.mkdir(parents=True, exist_ok=True)
        if fmt=="parquet": df.to_parquet(fn,index=False)
        else:              df.to_csv(fn,index=False)
        log.info(f"saved → {fn}")
        return fn


# =====================  EXAMPLE USAGE  =========================
#if __name__ == "__main__":
#    oc = OmniCollector()

#    csv_df   = oc.from_file("data/raw/users.csv")
#    s3_df    = oc.from_file("s3://my-bkt/2025/05/events.parquet")
#    pg_df    = oc.from_sql("postgresql://user:pwd@host/db",
                           "SELECT * FROM public.events LIMIT 1000")
#    api_df   = oc.from_rest("https://api.exchangerate.host/latest")

#    kafka_df = oc.from_kafka("tx-events","localhost:9092",batch=50)
#    sheet_df = oc.from_gsheet(sheet_key=os.getenv("GS_KEY"),
                              creds_json="gcp-creds.json")

#    oc.save(csv_df,"users")
#    oc.save(api_df, "fxrates")
