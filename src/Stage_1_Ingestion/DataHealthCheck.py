import pandas as pd
import numpy as np
import scipy.stats as ss
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DataHealthCheck:
    """
    Run a battery of pre-training EDA checks on a DataFrame and produce an HTML report.
    """

    def __init__(self, df: pd.DataFrame,
                 target_col: str = None,
                 batch_col: str = None,
                 datetime_cols: list = None):
        self.df = df.copy()
        self.n_rows, self.n_cols = df.shape
        self.p = self.n_cols
        self.n = self.n_rows
        self.target_col = target_col
        self.batch_col = batch_col
        self.datetime_cols = datetime_cols or []
        self.results = {}

    def detect_dimensionality(self):
        ratio = self.p / self.n
        tag = "p≫n" if self.p > self.n else (
            "n≫p" if self.n > self.p else "p≈n")
        self.results['dimensionality'] = {
            'n_rows': self.n,
            'n_cols': self.p,
            'ratio': f"{self.p}/{self.n}={ratio:.2f}",
            'regime': tag
        }

    def detect_missingness(self):
        miss = self.df.isna().mean().sort_values(ascending=False)
        self.results['missingness'] = {
            'overall_pct': miss.mean(),
            'top_missing_cols': miss.head(10).to_dict()
        }

    def detect_dtypes(self):
        counts = self.df.dtypes.value_counts().to_dict()
        self.results['dtypes'] = counts

    def detect_skew_scale(self):
        num = self.df.select_dtypes(include=[np.number])
        skew = num.skew().sort_values(ascending=False).head(10).to_dict()
        self.results['skewness'] = skew

    def detect_categorical_cardinality(self):
        cats = self.df.select_dtypes(include=['object', 'category'])
        card = {c: cats[c].nunique() for c in cats.columns}
        card = dict(
            sorted(card.items(), key=lambda kv: kv[1], reverse=True)[:10])
        self.results['cardinality'] = card

    def detect_outliers(self):
        out = {}
        num = self.df.select_dtypes(include=[np.number])
        for col in num.columns[:50]:  # limit to first 50 for speed
            q1, q3 = np.nanpercentile(num[col], [25, 75])
            iqr = q3 - q1
            low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            out[col] = int(((num[col] < low) | (num[col] > high)).sum())
        # top 10 outlier counts
        self.results['outliers'] = dict(
            sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:10])

    def detect_collinearity(self, thresh=0.9):
        num = self.df.select_dtypes(
            include=[np.number]).iloc[:, :200]  # cap for performance
        corr = num.corr().abs()
        pairs = [(i, j, corr.loc[i, j]) for i, j in combinations(
            corr.columns, 2) if corr.loc[i, j] > thresh]
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
        self.results['collinearity'] = [
            {'pair': (i, j), 'corr': f"{c:.2f}"} for i, j, c in pairs]

    def detect_vif(self):
        num = self.df.select_dtypes(
            include=[np.number]).dropna().iloc[:, :20]  # small subset
        X = num.values
        vifs = {num.columns[i]: variance_inflation_factor(
            X, i) for i in range(X.shape[1])}
        self.results['vif'] = dict(
            sorted(vifs.items(), key=lambda kv: kv[1], reverse=True)[:10])

    def detect_imbalance(self):
        if self.target_col and self.target_col in self.df:
            vc = self.df[self.target_col].value_counts(
                normalize=True).to_dict()
            self.results['imbalance'] = vc

    def detect_date_issues(self):
        info = {}
        for col in self.datetime_cols:
            if col in self.df:
                dates = pd.to_datetime(self.df[col], errors='coerce')
                info[col] = {
                    'parsed_pct': dates.notna().mean(),
                    'min': str(dates.min()),
                    'max': str(dates.max())
                }
        self.results['date_issues'] = info

    def detect_batch_summary(self):
        if self.batch_col and self.batch_col in self.df:
            vc = self.df[self.batch_col].value_counts(normalize=True).to_dict()
            self.results['batch_distribution'] = vc

    def run_all_checks(self):
        self.detect_dimensionality()
        self.detect_missingness()
        self.detect_dtypes()
        self.detect_skew_scale()
        self.detect_categorical_cardinality()
        self.detect_outliers()
        self.detect_collinearity()
        self.detect_vif()
        self.detect_imbalance()
        self.detect_date_issues()
        self.detect_batch_summary()

    def generate_report(self) -> str:
        self.run_all_checks()
        html = ['<html><head><style>',
                'body{font-family:sans-serif;padding:20px;}',
                'h2{border-bottom:1px solid #ddd;}',
                'table{border-collapse:collapse;margin-bottom:20px;}',
                'th,td{border:1px solid #ccc;padding:4px 8px;}',
                '</style></head><body>']
        html.append(f"<h1>DataFrame Health Report</h1>")
        # Dimensionality
        d = self.results['dimensionality']
        html.append("<h2>1. Dimensionality</h2>")
        html.append(
            f"<p>Rows: {d['n_rows']}, Columns: {d['n_cols']} &nbsp; (<strong>{d['regime']}</strong>, ratio={d['ratio']})</p>")

        # Missingness
        m = self.results['missingness']
        html.append("<h2>2. Missingness</h2>")
        html.append(f"<p>Overall missing: {m['overall_pct']*100:.2f}%</p>")
        html.append("<table><tr><th>Column</th><th>% Missing</th></tr>")
        for col, pct in m['top_missing_cols'].items():
            html.append(f"<tr><td>{col}</td><td>{pct*100:.1f}%</td></tr>")
        html.append("</table>")

        # dtypes
        html.append(
            "<h2>3. Data Types</h2><table><tr><th>dtype</th><th>count</th></tr>")
        for dt, cnt in self.results['dtypes'].items():
            html.append(f"<tr><td>{dt}</td><td>{cnt}</td></tr>")
        html.append("</table>")

        # Skewness
        html.append(
            "<h2>4. Top Skewed Numeric Features</h2><table><tr><th>Feature</th><th>Skew</th></tr>")
        for col, sk in self.results['skewness'].items():
            html.append(f"<tr><td>{col}</td><td>{sk:.2f}</td></tr>")
        html.append("</table>")

        # Cardinality
        html.append(
            "<h2>5. Categorical Cardinality (Top 10)</h2><table><tr><th>Feature</th><th>#Levels</th></tr>")
        for col, lvl in self.results['cardinality'].items():
            html.append(f"<tr><td>{col}</td><td>{lvl}</td></tr>")
        html.append("</table>")

        # Outliers
        html.append(
            "<h2>6. Univariate Outlier Counts (Top 10)</h2><table><tr><th>Feature</th><th>Outliers</th></tr>")
        for col, cnt in self.results['outliers'].items():
            html.append(f"<tr><td>{col}</td><td>{cnt}</td></tr>")
        html.append("</table>")

        # Collinearity
        html.append(
            "<h2>7. Strongly Correlated Pairs (r>0.9)</h2><table><tr><th>Pair</th><th>Correlation</th></tr>")
        for rec in self.results['collinearity']:
            html.append(
                f"<tr><td>{rec['pair']}</td><td>{rec['corr']}</td></tr>")
        html.append("</table>")

        # VIF
        html.append(
            "<h2>8. VIF (Top 10)</h2><table><tr><th>Feature</th><th>VIF</th></tr>")
        for col, v in self.results['vif'].items():
            html.append(f"<tr><td>{col}</td><td>{v:.2f}</td></tr>")
        html.append("</table>")

        # Imbalance
        if 'imbalance' in self.results:
            html.append(
                "<h2>9. Target Imbalance</h2><table><tr><th>Class</th><th>Pct</th></tr>")
            for cls, pct in self.results['imbalance'].items():
                html.append(f"<tr><td>{cls}</td><td>{pct*100:.1f}%</td></tr>")
            html.append("</table>")

        # Date issues
        if 'date_issues' in self.results:
            html.append("<h2>10. Date Columns</h2><table><tr><th>Column</th><th>Parsed %</th>"
                        "<th>Min</th><th>Max</th></tr>")
            for col, info in self.results['date_issues'].items():
                html.append(f"<tr><td>{col}</td><td>{info['parsed_pct']*100:.1f}%</td>"
                            f"<td>{info['min']}</td><td>{info['max']}</td></tr>")
            html.append("</table>")

        # Batch summary
        if 'batch_distribution' in self.results:
            html.append(
                "<h2>11. Batch Distribution</h2><table><tr><th>Batch</th><th>Pct</th></tr>")
            for b, pct in self.results['batch_distribution'].items():
                html.append(f"<tr><td>{b}</td><td>{pct*100:.1f}%</td></tr>")
            html.append("</table>")

        html.append("</body></html>")
        return "\n".join(html)
