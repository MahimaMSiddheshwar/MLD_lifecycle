## ✅ **How to Combine `PipelineReporter`, `PerfMixin`, and `monitor`**

### 🔹 1. **For ML pipeline components (like `OutlierDetector`, `MissingImputer`, etc.)**

Decorate the class using:

```python
from perfkit import PerfMixin

@perfclass(price_per_min=0.02)  # Optional cost tracking
class OutlierDetector(PerfMixin):
    ...
```

This gives:

- ⏱️ Wall-clock timing
- 🧠 Memory tracking
- 💰 Cost estimation
- ⚙️ Auto parallelism
- ⚡ GPU-acceleration (if enabled)

---

### 🔹 2. **For pipeline step functions (e.g., in ZenML)**

Decorate the step with `@monitor`:

```python
from utils.monitor import monitor

@monitor(name="outlier_detection_step", track_memory=True, track_input_size=True)
def outlier_step(data: pd.DataFrame) -> pd.DataFrame:
    detector = OutlierDetector(...)
    detector.fit(data)
    return detector.transform(data)
```

This gives:

- 🎯 Retry logic
- 🧠 Input/output logging
- 📈 MLflow model and report tracking
- 🪵 Pipeline log output to `pipeline_monitor.log`

---

### 🔹 3. **For final reporting (end of pipeline or after key components)**

Create and use the `PipelineReporter`:

```python
from reporting.pipeline_reporter import PipelineReporter

reporter = PipelineReporter(max_charts=50)

# Register components with `.report` or `.get_pipeline_report()` method
reporter.register("missing_imputer", missing_imputer)
reporter.register("outlier_detector", outlier_detector)

# Generate Markdown + HTML + JSON reports and log to MLflow
report = reporter.generate_report(output_name="final_pipeline_report")
```

This gives:

- 📊 Diagnostic visualizations (charts, top-k columns)
- 📄 Markdown + HTML + JSON reports
- 📦 MLflow artifact uploads
- 🧪 Universal step extensibility

---

## 🧩 Final Glue: ZenML Step

To fully integrate in **ZenML**, your step could look like:

```python
from zenml import step
from utils.monitor import monitor

@step
@monitor(name="preprocessing_step", track_memory=True, track_input_size=True)
def preprocessing_step(data: pd.DataFrame) -> pd.DataFrame:
    detector = OutlierDetector(...)
    detector.fit(data)
    df_clean = detector.transform(data)

    reporter = PipelineReporter()
    reporter.register("outliers", detector)
    reporter.generate_report(output_name="preprocessing_report")

    return df_clean
```

---

## 🚀 Summary

| Layer                   | Tool               | Features Provided                                    |
| ----------------------- | ------------------ | ---------------------------------------------------- |
| **Inside your class**   | `PerfMixin`        | Memory, timing, cost, parallel, GPU                  |
| **At step wrapper**     | `@monitor`         | Retry, input/output, MLflow, logs                    |
| **At pipeline summary** | `PipelineReporter` | Charts, top-k, .md/.json/.html, MLflow, perf summary |

Use them **together**, and your pipeline will be:
✔️ Tracked
✔️ Profiled
✔️ Visualized
✔️ Scalable
✔️ Production-ready
