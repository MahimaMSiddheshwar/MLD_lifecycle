import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, Union, Optional
import logging
import os

try:
    import mlflow
except ImportError:
    mlflow = None

log = logging.getLogger("PipelineReporter")
log.setLevel(logging.INFO)


class PipelineReporter:
    """
    World-class reporting utility for ML pipeline diagnostics.
    - Supports OutlierDetector, MissingImputer, and any compatible class.
    - Outputs Markdown + JSON + interactive HTML reports.
    - Integrates with MLflow, supports PerfMixin summary.
    """

    def __init__(
        self,
        max_charts: int = 50,
        report_dir: Union[str, Path] = "reports",
        enable_mlflow: bool = True
    ):
        self.max_charts = max_charts
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.report_dir / "figures"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.enable_mlflow = enable_mlflow
        self.components: Dict[str, Any] = {}

    def register(self, name: str, component: Any):
        """Register any compatible pipeline component."""
        self.components[name] = component

    def _plot_histogram(self, data: pd.Series, title: str, filename: str, hue: Optional[pd.Series] = None) -> str:
        plt.figure(figsize=(6, 4))
        if hue is not None:
            sns.histplot(data, hue=hue, kde=True, palette='muted')
        else:
            sns.histplot(data, kde=True, color='steelblue')
        plt.title(title)
        plt.tight_layout()
        filepath = self.plots_dir / filename
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def _handle_perf(self, component: Any, name: str) -> Optional[str]:
        if hasattr(component, "report"):
            perf_data = component.report()
            if perf_data:
                perf_file = self.report_dir / f"{name}_perf_summary.json"
                with open(perf_file, "w") as f:
                    json.dump(perf_data, f, indent=2)
                return str(perf_file)
        return None

    def _report_outlier_detector(self, name: str, component: Any) -> Dict:
        report = component.report if hasattr(component, "report") else {}
        outlier_indices = set(report.get(
            "real_outliers", {}).get("indices", []))
        df = getattr(component, "df", None)
        scores = getattr(component, "votes_table_",
                         {}).get("total_votes", None)
        cols = getattr(component, "numeric_cols", [])

        charts = []
        if df is not None and scores is not None:
            stds = {
                col: df[col].loc[list(outlier_indices)].std()
                for col in cols if col in df.columns
            }
            top_cols = sorted(stds.items(), key=lambda x: x[1], reverse=True)[
                :self.max_charts]
            for col, _ in top_cols:
                chart_path = self._plot_histogram(
                    df[col],
                    title=f"{name}: {col} (highlighted outliers)",
                    hue=df.index.isin(outlier_indices),
                    filename=f"{name}_{col}_outliers.png"
                )
                charts.append(chart_path)
        return {"charts": charts, "summary": report}

    def _report_missing_imputer(self, name: str, component: Any) -> Dict:
        report = component.report if hasattr(component, "report") else {}
        df = getattr(component, "df", None)
        cols = getattr(component, "numeric_cols", [])

        charts = []
        if df is not None:
            missing_frac = df[cols].isna().mean().sort_values(ascending=False)
            top_cols = missing_frac.head(self.max_charts).index
            for col in top_cols:
                chart_path = self._plot_histogram(
                    df[col],
                    title=f"{name}: {col} (missing values)",
                    filename=f"{name}_{col}_missing.png"
                )
                charts.append(chart_path)
        return {"charts": charts, "summary": report}

    def _generic_component_report(self, name: str, component: Any) -> Dict:
        """Fallback for any component with a `.get_pipeline_report()` method."""
        if hasattr(component, "get_pipeline_report"):
            return component.get_pipeline_report(report_dir=self.report_dir)
        return {"summary": str(component)}

    def generate_report(self, output_name: str = "pipeline_report") -> Dict:
        final_report = {}
        markdown = ["# Pipeline Diagnostic Report\n"]

        for name, comp in self.components.items():
            cls_name = comp.__class__.__name__
            if "Outlier" in cls_name:
                section = self._report_outlier_detector(name, comp)
            elif "Imputer" in cls_name:
                section = self._report_missing_imputer(name, comp)
            else:
                section = self._generic_component_report(name, comp)

            markdown.append(f"## {name} ({cls_name})\n")
            if section.get("summary"):
                markdown.append(
                    "```json\n" + json.dumps(section["summary"], indent=2) + "\n```\n")
            for chart in section.get("charts", []):
                markdown.append(f"![{name}]({chart})\n")

            final_report[name] = section

            # Perf summary (if applicable)
            perf_file = self._handle_perf(comp, name)
            if perf_file:
                markdown.append(f"\n📊 **Perf Summary**: `{perf_file}`\n")

        # Final file outputs
        json_path = self.report_dir / f"{output_name}.json"
        md_path = self.report_dir / f"{output_name}.md"
        html_path = self.report_dir / f"{output_name}.html"

        with open(json_path, "w") as f:
            json.dump(final_report, f, indent=2)
        with open(md_path, "w") as f:
            f.write("\n".join(markdown))
        with open(html_path, "w") as f:
            f.write("<html><body>"
                    + "<br><hr><br>".join(markdown) + "</body></html>")

        if self.enable_mlflow and mlflow:
            mlflow.log_artifact(json_path)
            mlflow.log_artifact(md_path)
            mlflow.log_artifact(html_path)
            for name, sect in final_report.items():
                for chart_path in sect.get("charts", []):
                    mlflow.log_artifact(chart_path)

        return final_report
