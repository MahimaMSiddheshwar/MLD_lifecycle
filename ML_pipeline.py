#!/usr/bin/env python3
"""
master_pipeline.py

MasterPipeline orchestrates the six‐stage preprocessing flow:
  1) Data collection (Stage 1)
  2) Missing‐value imputation (Stage 2)
  3) Outlier detection & treatment (Stage 3)
  4) Scaling & extra transforms (Stage 4)
  5) Categorical encoding (Stage 5)
  6) PCA reduction (Stage 6)

Usage:
    from master_pipeline import MasterPipeline
    import pandas as pd

    # Option A: supply a DataFrame directly
    df = pd.read_csv("data/raw/input_data.csv")
    pipeline = MasterPipeline(target_column="is_churn", random_state=42)
    (train_final, test_final) = pipeline.fit_transform(df)

    # Option B: read from CSV/Parquet paths using Stage 1 internally
    pipeline = MasterPipeline(target_column="is_churn", random_state=42)
    (train_final, test_final) = pipeline.fit_transform(
        source_type="csv", source_path="data/raw/input_data.csv"
    )

    # At any point, you can inspect pipeline.report to see diagnostics for each stage.
"""

from stage6_pca import Stage6PCA
from stage5_encoding import Stage5Encoder
from stage4_scaling_transform import Stage4Transform
from stage3_outlier import Stage3Outliers
from stage2_imputation import Stage2Imputer
from stage1_data_collection import DataCollector
import os
import pandas as pd

# ─── Stage 1: Data Collection ─────────────────────────────────────
from stage1_data_collection import OmniCollector

# ─── Stage 2: Missing‐Value Imputation ────────────────────────────
from stage2_imputation import MissingValueImputer

# ─── Stage 3: Outlier Detection & Treatment ──────────────────────
from stage3_outlier import OutlierDetector

# ─── Stage 4: Scaling & Extra Transformation ─────────────────────
from stage4_scaling_transform import ScalerTransformer

# ─── Stage 5: Categorical Encoding Variants ──────────────────────
from stage5_encoding import CategoricalEncoder

# ─── Stage 6: PCA Reduction ──────────────────────────────────────
from stage6_pca import PCAReducer


class MasterPipeline:
    """
    MasterPipeline stitches together six preprocessing stages. It can accept either:
      - A pandas DataFrame directly, or
      - A (source_type, source_path) pair to read via Stage 1.

    Each stage’s diagnostics are stored in pipeline.report under the corresponding key.
    """

    def __init__(
        self,
        *,
        target_column: str = None,
        test_size: float = 0.2,
        random_state: int = 42,
        pca_variance_threshold: float = 0.90,
        apply_pca: bool = True,
        verbose: bool = True
    ):
        # Shared settings
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.pca_variance_threshold = pca_variance_threshold
        self.apply_pca = apply_pca
        self.verbose = verbose

        # Instantiate each stage with matching parameters (where applicable)
        # Stage 1: OmniCollector does ingestion + semantic profiling
        self.collector = OmniCollector(
            pii_mask=True,
            validate=False,       # assume validation is done separately
            suite_name="default"  # if one uses Great Expectations
        )

        # Stage 2: Missing‐Value Imputer
        self.imputer = MissingValueImputer(
            target_column=self.target_column,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Stage 3: Outlier Detector
        self.outlier_detector = OutlierDetector(
            univ_iqr_factor=1.5,
            univ_zscore_cutoff=3.0,
            univ_modz_cutoff=3.5,
            multi_ci=0.975
        )

        # Stage 4: Scaler & Transform
        self.scaler_transformer = ScalerTransformer(
            transform_candidates=["none", "boxcox", "yeo", "quantile"],
            shapiro_n=5000,  # subsample size for Shapiro
            skew_thresh_standard=0.5,
            skew_thresh_robust=1.0,
            kurt_thresh_robust=5.0
        )

        # Stage 5: Categorical Encoder
        self.encoder = CategoricalEncoder(
            onehot_max_uniq=0.05,
            ordinal_max_uniq=0.20,
            knownally_max_uniq=0.50,
            ultrahigh_uniq=0.50
        )

        # Stage 6: PCA Reducer
        self.pca_reducer = PCAReducer(
            variance_threshold=self.pca_variance_threshold
        )

        # Master report aggregates each stage’s report under its own key
        self.report = {
            "stage1_ingestion": {},
            "stage2_imputation": {},
            "stage3_outlier": {},
            "stage4_scaling_transform": {},
            "stage5_encoding": {},
            "stage6_pca": {}
        }

    def fit_transform(
        self,
        X: pd.DataFrame = None,
        *,
        source_type: str = None,
        source_path: str = None,
        **source_kwargs
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Orchestrates all six stages on the provided data. Two usage patterns:

        1) If X (DataFrame) is given:
             train_df, test_df = pipeline.fit_transform(X)

        2) If you want to read from disk via Stage 1:
             train_df, test_df = pipeline.fit_transform(
                 source_type="csv", source_path="data/raw/data.csv"
             )

        Returns:
            processed_train_df, processed_test_df
        """

        # ── Stage 1: Data Collection ───────────────────────────
        if X is not None:
            # User provided a DataFrame; skip file‐reading logic
            raw_df = X.copy().reset_index(drop=True)
            if self.verbose:
                print("▶ Stage 1: Skipping file ingestion; DataFrame provided directly.")
        else:
            if source_type is None or source_path is None:
                raise ValueError(
                    "Either `X` must be a DataFrame or `source_type`+`source_path` must be provided."
                )
            if self.verbose:
                print(
                    f"▶ Stage 1: Ingesting data from ({source_type}) '{source_path}' …")
            # Use OmniCollector.readData(...) for file ingestion
            raw_df = self.collector.readData(
                source_type.lower(), source_path, **source_kwargs
            )
        # Stage 1 report: copy whatever OmniCollector logged (if reading from file)
        self.report["stage1_ingestion"] = getattr(self.collector, "report", {})

        # ── Stage 2: Missing‐Value Imputation ────────────────────
        if self.verbose:
            print("▶ Stage 2: Imputing missing values …")
        (train_df, test_df) = self.imputer.fit_transform(raw_df)
        self.report["stage2_imputation"] = self.imputer.report

        # ── Stage 3: Outlier Detection & Treatment ──────────────
        if self.verbose:
            print("▶ Stage 3: Detecting and treating outliers …")
        # apply outlier detection on train; transform test the same way
        train_df_clean = self.outlier_detector.fit_transform(train_df)
        test_df_clean = self.outlier_detector.transform(test_df)
        self.report["stage3_outlier"] = self.outlier_detector.report

        # ── Stage 4: Scaling & Extra Transformation ─────────────
        if self.verbose:
            print("▶ Stage 4: Scaling numeric columns and conditional transforms …")
        train_df_scaled = self.scaler_transformer.fit_transform(train_df_clean)
        test_df_scaled = self.scaler_transformer.transform(test_df_clean)
        self.report["stage4_scaling_transform"] = self.scaler_transformer.report

        # ── Stage 5: Categorical Encoding Variants ──────────────
        if self.verbose:
            print("▶ Stage 5: Encoding categorical features …")
        # Stage 5.encode_train returns only the "linear" variant by default.
        # We can capture all three variants on train, then transform test accordingly.
        df_lin_train = self.encoder.encode_train(train_df_scaled)
        df_lin_test = self.encoder.encode_test(test_df_scaled)
        # (If you need "tree" or "knn" variants, you can call
        #  encode_train(tree=True) or encode_train(knn=True) explicitly.)
        self.report["stage5_encoding"] = self.encoder.report

        # ── Stage 6: PCA Reduction ──────────────────────────────
        if self.apply_pca:
            if self.verbose:
                print("▶ Stage 6: Applying PCA on numeric features …")
            df_pca_train = self.pca_reducer.apply_pca(df_lin_train)
            df_pca_test = self.pca_reducer.transform(df_lin_test)
            self.report["stage6_pca"] = self.pca_reducer.report
        else:
            if self.verbose:
                print("▶ Stage 6: Skipping PCA (apply_pca=False).")
            df_pca_train = df_lin_train.copy()
            df_pca_test = df_lin_test.copy()
            self.report["stage6_pca"] = {
                "note": "PCA skipped by configuration."}

        # ── Return final processed train/test DataFrames ────────
        return df_pca_train, df_pca_test

    def get_report(self) -> dict:
        """
        Returns a nested dictionary of each stage’s internal `report` objects.
        """
        return self.report

    def print_summary(self) -> None:
        """
        Prints a concise summary of each stage’s key decisions.
        """
        print("\n" + "=" * 80)
        print("▶ Master Pipeline Report Summary")
        print("=" * 80)
        for stage, rpt in self.report.items():
            print(f"\n--- {stage.upper()} ---")
            if isinstance(rpt, dict):
                for key, val in rpt.items():
                    print(f"{key}: {val}")
            else:
                print(rpt)
        print("\n" + "=" * 80 + "\n")


# If someone runs this file directly, demonstrate usage with a toy DataFrame.
if __name__ == "__main__":
    import numpy as np

    # Create a small synthetic DataFrame
    data = {
        "feature_num1": [1.0, 2.5, np.nan, 4.0, 5.0, 1000.0],
        "feature_num2": [10, np.nan, 30, 40, 50, 60],
        "feature_cat": ["A", "B", "A", None, "C", "B"],
        "is_churn": [0, 1, 0, 1, 0, 1]
    }
    df_demo = pd.DataFrame(data)

    pipeline = MasterPipeline(
        target_column="is_churn",
        test_size=0.2,
        random_state=42,
        apply_pca=True,
        verbose=True
    )
    train_final, test_final = pipeline.fit_transform(df_demo)
    pipeline.print_summary()

    print("Final train shape:", train_final.shape)
    print("Final test shape :", test_final.shape)


########### _#_#_#__##__#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
# End of master_pipeline.py
########### _#_#_#__##__#_#_#_#_#_#_#_#_#_#_#_#_#_#_#


#!/usr/bin/env python3
"""
master_pipeline.py

MasterPipeline orchestrates the six‐stage preprocessing flow:
  1) Data collection (Stage 1)
  2) Missing‐value imputation (Stage 2)
  3) Outlier detection & treatment (Stage 3)
  4) Scaling & extra transforms (Stage 4)
  5) Categorical encoding (Stage 5)
  6) PCA reduction (Stage 6)

Usage:
    from master_pipeline import MasterPipeline
    import pandas as pd

    # Option A: supply a DataFrame directly
    df = pd.read_csv("data/raw/input_data.csv")
    pipeline = MasterPipeline(target_column="is_churn", random_state=42)
    (train_final, test_final) = pipeline.fit_transform(df)

    # Option B: read from CSV/Parquet paths using Stage 1 internally
    pipeline = MasterPipeline(target_column="is_churn", random_state=42)
    (train_final, test_final) = pipeline.fit_transform(
        source_type="csv", source_path="data/raw/input_data.csv"
    )

    # At any point, you can inspect pipeline.report to see diagnostics for each stage.
"""


# ─── Stage 1: Data Collection ─────────────────────────────────────

# ─── Stage 2: Missing‐Value Imputation ────────────────────────────

# ─── Stage 3: Outlier Detection & Treatment ──────────────────────

# ─── Stage 4: Scaling & Extra Transformation ─────────────────────

# ─── Stage 5: Categorical Encoding Variants ──────────────────────

# ─── Stage 6: PCA Reduction ──────────────────────────────────────


class MasterPipeline:
    """
    MasterPipeline stitches together six preprocessing stages. It can accept either:
      - A pandas DataFrame directly, or
      - A (source_type, source_path) pair to read via Stage 1.

    Each stage’s diagnostics are stored in pipeline.report under the corresponding key.
    """

    def __init__(
        self,
        *,
        target_column: str = None,
        test_size: float = 0.2,
        random_state: int = 42,
        pca_variance_threshold: float = 0.90,
        apply_pca: bool = True,
        verbose: bool = True
    ):
        # Shared settings
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.pca_variance_threshold = pca_variance_threshold
        self.apply_pca = apply_pca
        self.verbose = verbose

        # Instantiate each stage with matching parameters (where applicable)
        # Stage 1: OmniCollector does ingestion + semantic profiling
        self.collector = OmniCollector(
            pii_mask=True,
            validate=False,       # assume validation is done separately
            suite_name="default"  # if one uses Great Expectations
        )

        # Stage 2: Missing‐Value Imputer
        self.imputer = MissingValueImputer(
            target_column=self.target_column,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Stage 3: Outlier Detector
        self.outlier_detector = OutlierDetector(
            univ_iqr_factor=1.5,
            univ_zscore_cutoff=3.0,
            univ_modz_cutoff=3.5,
            multi_ci=0.975
        )

        # Stage 4: Scaler & Transform
        self.scaler_transformer = ScalerTransformer(
            transform_candidates=["none", "boxcox", "yeo", "quantile"],
            shapiro_n=5000,  # subsample size for Shapiro
            skew_thresh_standard=0.5,
            skew_thresh_robust=1.0,
            kurt_thresh_robust=5.0
        )

        # Stage 5: Categorical Encoder
        self.encoder = CategoricalEncoder(
            onehot_max_uniq=0.05,
            ordinal_max_uniq=0.20,
            knownally_max_uniq=0.50,
            ultrahigh_uniq=0.50
        )

        # Stage 6: PCA Reducer
        self.pca_reducer = PCAReducer(
            variance_threshold=self.pca_variance_threshold
        )

        # Master report aggregates each stage’s report under its own key
        self.report = {
            "stage1_ingestion": {},
            "stage2_imputation": {},
            "stage3_outlier": {},
            "stage4_scaling_transform": {},
            "stage5_encoding": {},
            "stage6_pca": {}
        }

    def fit_transform(
        self,
        X: pd.DataFrame = None,
        *,
        source_type: str = None,
        source_path: str = None,
        **source_kwargs
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Orchestrates all six stages on the provided data. Two usage patterns:

        1) If X (DataFrame) is given:
             train_df, test_df = pipeline.fit_transform(X)

        2) If you want to read from disk via Stage 1:
             train_df, test_df = pipeline.fit_transform(
                 source_type="csv", source_path="data/raw/data.csv"
             )

        Returns:
            processed_train_df, processed_test_df
        """

        # ── Stage 1: Data Collection ───────────────────────────
        if X is not None:
            # User provided a DataFrame; skip file‐reading logic
            raw_df = X.copy().reset_index(drop=True)
            if self.verbose:
                print("▶ Stage 1: Skipping file ingestion; DataFrame provided directly.")
        else:
            if source_type is None or source_path is None:
                raise ValueError(
                    "Either `X` must be a DataFrame or `source_type`+`source_path` must be provided."
                )
            if self.verbose:
                print(
                    f"▶ Stage 1: Ingesting data from ({source_type}) '{source_path}' …")
            # Use OmniCollector.readData(...) for file ingestion
            raw_df = self.collector.readData(
                source_type.lower(), source_path, **source_kwargs
            )
        # Stage 1 report: copy whatever OmniCollector logged (if reading from file)
        self.report["stage1_ingestion"] = getattr(self.collector, "report", {})

        # ── Stage 2: Missing‐Value Imputation ────────────────────
        if self.verbose:
            print("▶ Stage 2: Imputing missing values …")
        (train_df, test_df) = self.imputer.fit_transform(raw_df)
        self.report["stage2_imputation"] = self.imputer.report

        # ── Stage 3: Outlier Detection & Treatment ──────────────
        if self.verbose:
            print("▶ Stage 3: Detecting and treating outliers …")
        # apply outlier detection on train; transform test the same way
        train_df_clean = self.outlier_detector.fit_transform(train_df)
        test_df_clean = self.outlier_detector.transform(test_df)
        self.report["stage3_outlier"] = self.outlier_detector.report

        # ── Stage 4: Scaling & Extra Transformation ─────────────
        if self.verbose:
            print("▶ Stage 4: Scaling numeric columns and conditional transforms …")
        train_df_scaled = self.scaler_transformer.fit_transform(train_df_clean)
        test_df_scaled = self.scaler_transformer.transform(test_df_clean)
        self.report["stage4_scaling_transform"] = self.scaler_transformer.report

        # ── Stage 5: Categorical Encoding Variants ──────────────
        if self.verbose:
            print("▶ Stage 5: Encoding categorical features …")
        # Stage 5.encode_train returns only the "linear" variant by default.
        # We can capture all three variants on train, then transform test accordingly.
        df_lin_train = self.encoder.encode_train(train_df_scaled)
        df_lin_test = self.encoder.encode_test(test_df_scaled)
        # (If you need "tree" or "knn" variants, you can call
        #  encode_train(tree=True) or encode_train(knn=True) explicitly.)
        self.report["stage5_encoding"] = self.encoder.report

        # ── Stage 6: PCA Reduction ──────────────────────────────
        if self.apply_pca:
            if self.verbose:
                print("▶ Stage 6: Applying PCA on numeric features …")
            df_pca_train = self.pca_reducer.apply_pca(df_lin_train)
            df_pca_test = self.pca_reducer.transform(df_lin_test)
            self.report["stage6_pca"] = self.pca_reducer.report
        else:
            if self.verbose:
                print("▶ Stage 6: Skipping PCA (apply_pca=False).")
            df_pca_train = df_lin_train.copy()
            df_pca_test = df_lin_test.copy()
            self.report["stage6_pca"] = {
                "note": "PCA skipped by configuration."}

        # ── Return final processed train/test DataFrames ────────
        return df_pca_train, df_pca_test

    def get_report(self) -> dict:
        """
        Returns a nested dictionary of each stage’s internal `report` objects.
        """
        return self.report

    def print_summary(self) -> None:
        """
        Prints a concise summary of each stage’s key decisions.
        """
        print("\n" + "=" * 80)
        print("▶ Master Pipeline Report Summary")
        print("=" * 80)
        for stage, rpt in self.report.items():
            print(f"\n--- {stage.upper()} ---")
            if isinstance(rpt, dict):
                for key, val in rpt.items():
                    print(f"{key}: {val}")
            else:
                print(rpt)
        print("\n" + "=" * 80 + "\n")


# If someone runs this file directly, demonstrate usage with a toy DataFrame.
if __name__ == "__main__":
    import numpy as np

    # Create a small synthetic DataFrame
    data = {
        "feature_num1": [1.0, 2.5, np.nan, 4.0, 5.0, 1000.0],
        "feature_num2": [10, np.nan, 30, 40, 50, 60],
        "feature_cat": ["A", "B", "A", None, "C", "B"],
        "is_churn": [0, 1, 0, 1, 0, 1]
    }
    df_demo = pd.DataFrame(data)

    pipeline = MasterPipeline(
        target_column="is_churn",
        test_size=0.2,
        random_state=42,
        apply_pca=True,
        verbose=True
    )
    train_final, test_final = pipeline.fit_transform(df_demo)
    pipeline.print_summary()

    print("Final train shape:", train_final.shape)
    print("Final test shape :", test_final.shape)
########### _#_#_#__##__#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
# End of master_pipeline.py
########### _#_#_#__##__#_#_#_#_#_#_#_#_#_#_#_#_#_#_#


#!/usr/bin/env python3
"""
Master script to run Stages 1→6 in order.
"""


# 1) Data Collection (Stage 1)
collector = DataCollector(pii_mask=True, validate=False)
df_raw = collector.read_flatfile(
    "data/raw/your_file.csv")  # or any other source

# 2) Missing‐Value Imputation (Stage 2)
imputer = Stage2Imputer(max_missing_frac_drop=0.9,
                        knn_neighbors=5, cat_tvd_cutoff=0.2)
df_imp = imputer.fit_transform(df_raw)

# 3) Outlier Detection & Winsorization (Stage 3)
outlier_stage = Stage3Outliers()
df_out = outlier_stage.fit_transform(df_imp)

# 4) Scaling & Extra Transform (Stage 4)
transformer = Stage4Transform()
df_scaled = transformer.fit_transform(df_out)

# 5) Categorical Encoding (Stage 5)
encoder = Stage5Encoder(onehot_frac_thresh=0.05,
                        ordinal_frac_thresh=0.20, freq_frac_thresh=0.50)
df_encoded = encoder.fit_transform(df_scaled)

# 6) PCA (Stage 6)
pca_stage = Stage6PCA(variance_threshold=0.90, cond_thresh=1e6)
df_final = pca_stage.fit_transform(df_encoded)

# df_final is now ready for modeling (Stages 7+).
# All intermediate reports and plots have been saved under 'reports/'.
