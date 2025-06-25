import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.Stage_4_Preprocessor.Outlier_Detection import OutlierDetector
from src.Stage_4_Preprocessor.Missing_Imputer import MissingImputer

DATASET_TARGET_COLUMN_NAME = "label"


@step
def missing_imputer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"], Annotated[pd.DataFrame, "test"]]:

    imputer = MissingImputer(
        max_missing_frac_drop=0.8,
        knn_neighbors=2,
        knn_mice_max_rows=10,
        knn_mice_max_columns=3,
        var_ratio_cutoff=0.5,
        cov_change_cutoff=0.2,
        rare_freq_cutoff=0.1,
        random_state=0,
        verbose=True
    )
    train_imp = imputer.fit_transform(train)
    test_imp = imputer.transform(test)
    val_imp = imputer.transform(val)

    print("\nImputed Training DataFrame:")
    print("train_imp", train_imp.shape)
    print("test_imp", test_imp.shape)
    print("val_imp", val_imp.shape)
    print("\nDropped columns:", imputer.cols_to_drop)
    print("Numeric columns:", imputer.numeric_cols)
    print("Categorical columns:", imputer.categorical_cols)
    print("Numeric imputers:", imputer.numeric_imputers)
    print("Categorical imputers:", imputer.categorical_imputers)

    return train_imp, test_imp, val_imp


@step
def outlier_imputer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"], Annotated[pd.DataFrame, "test"]]:

    detector = OutlierDetector(
        outlier_threshold=3,
        robust_covariance=True,
        cap_outliers=None,       # will force winsorize unless cap_outliers=False
        model_family="linear",
        random_state=0,
        verbose=True
    )

    train_clean = detector.fit_transform(train)

    print(detector.report["univariate_outliers"])
    # chosen method, etc.
    print(detector.report["multivariate_outliers"])
    print(detector.report["real_outliers"])
    # what was done: drop/winsorize
    print(detector.report["treatment"])
    # see each row’s votes
    print(detector.votes_table_.head())
    # how many clipped per column
    print(detector.clipped_counts_)

    # 4) On new (validation/test) data, simply flag:
    test_clean = detector.transform(test)
    test_outliers = detector.outlier_flags_

    print("Test Outliers:", test_outliers.sum(),
          "out of", len(test_clean).sum())

    val_clean = detector.transform(val)
    val_outliers = detector.outlier_flags_
    print("Test Outliers:", val_outliers.sum(),
          "out of", len(val_clean).sum())
    return train_clean, test_clean, val_clean


@step
def data_preprocessor(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"], Annotated[pd.DataFrame, "test"]]:
    if not DATASET_TARGET_COLUMN_NAME:
        raise ValueError(
            "DATASET_TARGET_COLUMN_NAME must be set to generate a baseline model.")
    # TODO: create 2 seperate dataframe of threewaysplit for linear and tree based models
    train, test, val = missing_imputer(train, test, val)
    train, test, val = outlier_imputer(train, test, val)
    return train, test, val
