import pandas as pd
import numpy as np


class LeakageDetection:
    """
    Section 4.5: Leakage Detection.
    Checks for potential data leakage issues such as target leakage in features or improper data splits.
    """

    def __init__(self):
        self.leakage_report = {}

    def check(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame = None, y_test: pd.Series = None):
        """
        Perform checks for data leakage.
        Returns a dict with findings.
        """
        report = {}
        # Target leakage check: features too correlated with target
        if not y_train.empty:
            if (y_train.dtype in [float, int]) and (y_train.nunique() > 2):
                # Regression target
                corr_with_target = {}
                for col in X_train.select_dtypes(include=np.number).columns:
                    if X_train[col].var() == 0:
                        continue
                    corr = np.corrcoef(X_train[col], y_train)[0, 1]
                    if abs(corr) > 0.99:
                        corr_with_target[col] = corr
                if corr_with_target:
                    report['high_corr_features_target'] = corr_with_target
            else:
                # Classification target
                potential_leaks = {}
                for col in X_train.columns:
                    data = pd.DataFrame({col: X_train[col], 'target': y_train})
                    if data[col].dtype.kind in 'if' and data[col].nunique() > 10:
                        data['bin'] = pd.qcut(data[col], q=min(
                            10, len(data)), duplicates='drop')
                        groups = data.groupby('bin')['target']
                    else:
                        groups = data.groupby(col)['target']
                    for val, grp in groups:
                        if len(grp) > 0 and grp.nunique(dropna=False) == 1:
                            potential_leaks.setdefault(col, []).append(val)
                if potential_leaks:
                    report['potential_leakage_features'] = potential_leaks

        # Overlap leakage: check if any identical feature rows exist in both train and test
        if X_test is not None:
            common = pd.merge(X_train, X_test, how='inner')
            if not common.empty:
                report['overlap_rows_train_test'] = len(common)

        # Unseen category check: if test has categories not present in train (not a leakage, but important for model)
        if X_test is not None:
            cat_cols = [col for col in X_train.columns if X_train[col].dtype == object or str(
                X_train[col].dtype).startswith('category')]
            unseen = {}
            for col in cat_cols:
                if col in X_test.columns:
                    train_vals = set(X_train[col].dropna().unique())
                    test_vals = set(X_test[col].dropna().unique())
                    diff = test_vals - train_vals
                    if diff:
                        unseen[col] = diff
            if unseen:
                report['unseen_test_categories'] = unseen

        self.leakage_report = report
        return report
