#!/usr/bin/env python3
"""
pipeline.py

Function-based ML pipeline for stages 0–6:
  0) ingest: Data ingestion to Parquet
  1) inspect: Data inspection and HTML report
  2) split: Train/Val/Test split to Parquet
  3) impute: Missing-value imputation + log
  4) scale_transform: Scaling/transform + log
  5) detect_outliers: Outlier detection + log
  6) encode: Categorical encoding + log

Each function reads/writes DataFrames and logs metadata for reproducibility.
"""

from typing import Optional, Tuple
import logging
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.Data_Ingest_diagnose.data_injection_stage1 import DataCollector
from src.Data_Ingest_diagnose.data_inspection import DataFrameHealthCheck
from src.Data_Preprocessing.improved_stage2 import Stage2Imputer
from src.Data_Preprocessing.scaling_transform_stage3 import Stage4Transform
from src.Data_Preprocessing.OutlierDetection_stage4 import OutlierDetector
from src.Feature_Engineering.encoding_stage5 import Stage5Encoder


def ingest(source, mode="file", output_dir="outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collector = DataCollector(pii_mask=True, validate=False)
    df = collector.read_file(
        source) if mode == "file" else collector.read_sql(source)
    path = output_dir/"0_raw.parquet"
    df.to_parquet(path, index=False)
    print(f"[INGEST] Raw data saved to {path}")
    return df


def inspect(df, target_column=None, output_dir="outputs"):
    output_dir = Path(output_dir)
    inspector = DataFrameHealthCheck(df, target_col=target_column)
    inspector.run_all_checks()
    report = output_dir/"1_inspection_report.html"
    inspector.generate_report(report)
    print(f"[INSPECT] Report saved to {report}")
    return df


def split(df, target_column=None, test_size=0.1, val_size=0.1,
          random_state=42, output_dir="outputs"):
    output_dir = Path(output_dir)
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=(df[target_column] if target_column else None)
    )
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size),
        random_state=random_state,
        stratify=(train_val[target_column] if target_column else None)
    )
    for name, subset in [("2_train", train), ("3_val", val), ("4_test", test)]:
        subset.to_parquet(output_dir/f"{name}.parquet", index=False)
        print(f"[SPLIT] {name} set saved")
    return train, val, test


def impute(train, val, test, random_state=42, output_dir="outputs"):
    output_dir = Path(output_dir)
    imputer = Stage2Imputer(random_state=random_state)
    imputer.fit(train)
    train_i = imputer.transform(train)
    val_i = imputer.transform(val)
    test_i = imputer.transform(test)
    for name, df_i in [("5_train_imputed", train_i),
                       ("6_val_imputed", val_i),
                       ("7_test_imputed", test_i)]:
        df_i.to_parquet(output_dir/f"{name}.parquet", index=False)
    with open(output_dir/"impute_log.json", "w") as f:
        json.dump(imputer.report, f, indent=2)
    print("[IMPUTE] Data imputed and log saved")
    return train_i, val_i, test_i


def scale_transform(train, val, test, output_dir="outputs"):
    output_dir = Path(output_dir)
    scaler = Stage4Transform()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    for idx, df_s in enumerate([train_s, val_s, test_s], start=8):
        df_s.to_parquet(output_dir/f"{idx}_scaled.parquet", index=False)
    with open(output_dir/"scale_log.json", "w") as f:
        json.dump(scaler.report, f, indent=2)
    print("[SCALE] Scaling complete and log saved")
    return train_s, val_s, test_s


def detect_outliers(train, val, test, output_dir="outputs"):
    output_dir = Path(output_dir)
    detector = OutlierDetector()
    train_o = detector.fit_transform(train)
    val_o = detector.transform(val)
    test_o = detector.transform(test)
    for idx, df_o in enumerate([train_o, val_o, test_o], start=11):
        df_o.to_parquet(output_dir/f"{idx}_clean.parquet", index=False)
    with open(output_dir/"outlier_log.json", "w") as f:
        json.dump(detector.report, f, indent=2)
    print("[OUTLIER] Outliers handled and log saved")
    return train_o, val_o, test_o


def encode(train, val, test, output_dir="outputs"):
    output_dir = Path(output_dir)
    encoder = Stage5Encoder()
    train_e = encoder.encode_train(train)
    val_e = encoder.encode_test(val)
    test_e = encoder.encode_test(test)
    for idx, df_e in enumerate([train_e, val_e, test_e], start=14):
        df_e.to_parquet(output_dir/f"{idx}_encoded.parquet", index=False)
    with open(output_dir/"encode_log.json", "w") as f:
        json.dump(encoder.report, f, indent=2)
    print("[ENCODE] Encoding done and log saved")
    return train_e, val_e, test_e


def run_all(source, mode="file", target_column=None,
            test_size=0.1, val_size=0.1, random_state=42,
            output_dir="outputs"):
    df = ingest(source, mode, output_dir)
    df = inspect(df, target_column, output_dir)
    train, val, test = split(df, target_column, test_size,
                             val_size, random_state, output_dir)
    train_i, val_i, test_i = impute(train, val, test, random_state, output_dir)
    train_s, val_s, test_s = scale_transform(
        train_i, val_i, test_i, output_dir)
    train_o, val_o, test_o = detect_outliers(
        train_s, val_s, test_s, output_dir)
    return encode(train_o, val_o, test_o, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Function-based ML pipeline")
    parser.add_argument("source", help="File path or SQL connection string")
    parser.add_argument("--mode", choices=["file", "sql"], default="file")
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--stage", choices=[
        "ingest", "inspect", "split", "impute",
        "scale_transform", "detect_outliers", "encode"
    ], help="Run a single stage")
    args = parser.parse_args()

    if args.all:
        run_all(args.source, args.mode, args.target,
                args.test_size, args.val_size,
                args.random_state, args.output)
    elif args.stage:
        # minimal single-stage runner (tweak as needed)
        from pathlib import Path
        import pandas as pd
        mapping = {
            "inspect": "0_raw.parquet",
            "split":   "0_raw.parquet",
            "impute":  "2_train.parquet",
            "scale_transform": "5_train_imputed.parquet",
            "detect_outliers": "8_train_scaled.parquet",
            "encode":  "11_train_clean.parquet"
        }
        if args.stage == "ingest":
            ingest(args.source, args.mode, args.output)
        else:
            df_in = pd.read_parquet(Path(args.output)/mapping[args.stage])
            func = globals()[args.stage]
            if args.stage == "split":
                func(df_in, args.target, args.test_size,
                     args.val_size, args.random_state, args.output)
            else:
                # for multi-input stages you'll load train/val/test manually
                func(df_in)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
# ==============================================================

#!/usr/bin/env python3
"""
pipeline.py

Function-based ML pipeline for stages 0–6 with argument validation,
logging, Parquet outputs, and JSON metadata reporting.
"""


# Configure module-level logger
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")


def ingest(source: str, mode: str = "file", output_dir: str = "outputs") -> pd.DataFrame:
    """
    Stage 0: Read raw data into a DataFrame and save to Parquet.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    collector = DataCollector(pii_mask=True, validate=False)
    if mode == "file":
        if not Path(source).exists():
            raise FileNotFoundError(f"Input file not found: {source}")
        df = collector.read_file(source)
    else:
        df = collector.read_sql(source)
    path = out / "0_raw.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"[INGEST] Raw data saved to {path}")
    return df


def inspect(df: pd.DataFrame, target_column: Optional[str] = None, output_dir: str = "outputs") -> pd.DataFrame:
    """
    Stage 1: Run health checks and output an HTML report.
    """
    out = Path(output_dir)
    inspector = DataFrameHealthCheck(df, target_col=target_column)
    inspector.run_all_checks()
    report = out / "1_inspection_report.html"
    inspector.generate_report(report)
    logger.info(f"[INSPECT] HTML report saved to {report}")
    return df


def split(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    output_dir: str = "outputs"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 2: Stratified train/val/test split.
    """
    _validate_sizes(test_size, val_size)
    out = Path(output_dir)
    stratify = df[target_column] if target_column else None
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state, stratify=(
        train_val[target_column] if target_column else None))
    for name, subset in (("2_train", train), ("3_val", val), ("4_test", test)):
        path = out / f"{name}.parquet"
        subset.to_parquet(path, index=False)
        logger.info(f"[SPLIT] {name} saved to {path}")
    return train, val, test


def impute(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    random_state: int = 42,
    output_dir: str = "outputs"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 3: Fit & apply missing-value imputation. Log parameters.
    """
    out = Path(output_dir)
    imputer = Stage2Imputer(random_state=random_state)
    imputer.fit(train)
    train_i = imputer.transform(train)
    val_i = imputer.transform(val)
    test_i = imputer.transform(test)
    for tag, df_i in (("5_train_imputed", train_i), ("6_val_imputed", val_i), ("7_test_imputed", test_i)):
        path = out / f"{tag}.parquet"
        df_i.to_parquet(path, index=False)
    with open(out / "impute_log.json", "w") as f:
        json.dump(imputer.report, f, indent=2)
    logger.info("[IMPUTE] Completed and logged")
    return train_i, val_i, test_i


def scale_transform(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = "outputs"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 4: Fit & apply scaling/transform. Log parameters.
    """
    out = Path(output_dir)
    scaler = Stage4Transform()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    for idx, df_s in enumerate((train_s, val_s, test_s), start=8):
        path = out / f"{idx}_scaled.parquet"
        df_s.to_parquet(path, index=False)
    with open(out / "scale_log.json", "w") as f:
        json.dump(scaler.report, f, indent=2)
    logger.info("[SCALE] Completed and logged")
    return train_s, val_s, test_s


def detect_outliers(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = "outputs"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 5: Fit & remove/cap outliers. Log stats.
    """
    out = Path(output_dir)
    detector = OutlierDetector()
    train_o = detector.fit_transform(train)
    val_o = detector.transform(val)
    test_o = detector.transform(test)
    for idx, df_o in enumerate((train_o, val_o, test_o), start=11):
        path = out / f"{idx}_clean.parquet"
        df_o.to_parquet(path, index=False)
    with open(out / "outlier_log.json", "w") as f:
        json.dump(detector.report, f, indent=2)
    logger.info("[OUTLIER] Completed and logged")
    return train_o, val_o, test_o


def encode(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = "outputs"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 6: Fit & apply categorical encoding. Log details.
    """
    out = Path(output_dir)
    encoder = Stage5Encoder()
    train_e = encoder.encode_train(train)
    val_e = encoder.encode_test(val)
    test_e = encoder.encode_test(test)
    for idx, df_e in enumerate((train_e, val_e, test_e), start=14):
        path = out / f"{idx}_encoded.parquet"
        df_e.to_parquet(path, index=False)
    with open(out / "encode_log.json", "w") as f:
        json.dump(encoder.report, f, indent=2)
    logger.info("[ENCODE] Completed and logged")
    return train_e, val_e, test_e


def run_all(
    source: str,
    mode: str = "file",
    target_column: Optional[str] = None,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    output_dir: str = "outputs"
):
    """
    Run the entire pipeline from ingestion through encoding.
    """
    df0 = ingest(source, mode, output_dir)
    df1 = inspect(df0, target_column, output_dir)
    t, v, tst = split(df1, target_column, test_size,
                      val_size, random_state, output_dir)
    ti, vi, tsi = impute(t, v, tst, random_state, output_dir)
    ts, vs, tss = scale_transform(ti, vi, tsi, output_dir)
    to, vo, tso = detect_outliers(ts, vs, tss, output_dir)
    return encode(to, vo, tso, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Industry-standard function-based ML pipeline")
    parser.add_argument("source", help="File path or SQL connection string")
    parser.add_argument("--mode", choices=["file", "sql"], default="file")
    parser.add_argument("--target", help="Target column name", default=None)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output", default="outputs",
                        help="Directory for outputs")
    parser.add_argument("--all", action="store_true",
                        help="Run all stages sequentially")
    parser.add_argument("--stage", choices=[
        "ingest", "inspect", "split", "impute",
        "scale_transform", "detect_outliers", "encode"
    ], help="Run only the specified stage")
    args = parser.parse_args()

    if args.all:
        run_all(
            source=args.source,
            mode=args.mode,
            target_column=args.target,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            output_dir=args.output
        )
    elif args.stage:
        # Single-stage invocation handled here as needed...
        parser.print_help()
    else:
        parser.print_help()
# ==================================÷===========================
"""
# Advanced Feature Splitter and Constructor Integration

```diff
--- pipeline.py
+++ pipeline.py


@@ imports
 from src.Feature_Engineering.encoding_stage5 import Stage5Encoder
+from src.advanced_splitting import AdvancedFeatureSplitterV4
+from src.advanced_construction import AdvancedFeatureConstructorV4


@@
-def run_all(
+def run_all(
     source: str,
     mode: str="file",
     target_column: Optional[str]=None,
     test_size: float=0.1,
     val_size: float=0.1,
     random_state: int=42,
     output_dir: str="outputs"
 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
@ @
     # Stage 5: Outlier detection
     to, vo, tso=detect_outliers(ts, vs, tss, output_dir)
+
+    # Stage 6: Advanced feature splitting
+ splitter=AdvancedFeatureSplitterV4(
+ n_jobs=4, cache_json=True, strip_html=True, flatten_xml=True,
+        # you can pass mixed_patterns=..., url_columns=[...], etc.
+)
+ train_fs=splitter.transform(to)
+ val_fs=splitter.transform(vo)
+ test_fs=splitter.transform(tso)
+    # Optionally log splitter.report_ to JSON
+ with open(Path(output_dir)/"splitter_report.json", "w") as f:
+ json.dump(splitter.report_, f, indent=2)
+
+    # Stage 7: Advanced feature construction
+ constructor=AdvancedFeatureConstructorV4(
+ n_jobs=4,
+        group_aggs={'user_id': ['mean', 'count']},
+        crosses=[('colA', 'colB')],
+        text_stat_cols=['comment'],
+        rolling_windows={'value': 5},
+        custom_funcs={'range': lambda df: df['max']-df['min']},
+)
+ train_fc=constructor.transform(train_fs)
+ val_fc=constructor.transform(val_fs)
+ test_fc=constructor.transform(test_fs)
+    # Optionally log constructor.report_ to JSON
+ with open(Path(output_dir)/"constructor_report.json", "w") as f:
+ json.dump(constructor.report_, f, indent=2)
+
+    # Stage 8: Encoding
+ return encode(train_fc, val_fc, test_fc, output_dir)
```

1. ** Imports**: Bring in `AdvancedFeatureSplitterV4` and `AdvancedFeatureConstructorV4`.
2. ** Within `run_all`**, after outlier detection but before encoding:

   * Instantiate and apply the splitter to each split DataFrame, logging its `report_`.
   * Instantiate and apply the constructor to the splitter outputs, logging its `report_`.
3. ** Rename ** subsequent `encode` stage to occur after construction.
"""
