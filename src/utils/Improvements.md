Below is a **concise backlog ✏️** of the _still-open_ performance ideas **# 5 → # 16** followed by a ready-to-drop **`README.md`** that shows exactly how to plug **perfkit v4** into any of your pipeline stages.

---

## ⏳ Pending optimisations (ideas 5 – 16)

| Idea # | Title                                  | Essence                                                                 | Where it lives                                  | Why it isn’t in perfkit v4 yet                    |
| -----: | -------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
|  **5** | _Blocked column processing_            | Batch very-wide matrices (e.g. 500 cols/block) to cap RAM & ℴ(p²) work. | Each stage that loops over `self.numeric_cols`. | Needs domain logic (covariance vs. box-cox etc.). |
|  **6** | _Early-exit skip rules_                | If a column is < 5 % NA → mean-impute immediately.                      | `Stage2Imputer` numeric loop.                   | Depends on your QC thresholds.                    |
|  **7** | _Adaptive algorithm menu_              | Skip KNN/MICE whenever `p ≫ n` or dataset too wide.                     | `Stage2Imputer` decision logic.                 | Already half-implemented but can be refined.      |
|  **8** | _Categorical dictionary encoding_      | Convert rare-collapsed strings to `category` → `int32`.                 | `Stage2Imputer` categorical section.            | Requires one-line dtype cast after rare-collapse. |
|  **9** | _Sparse frame support_                 | Auto-convert very sparse numeric blocks to CSR/CSC.                     | Any stage that holds TF-IDF / one-hot data.     | Needs sparse-aware stats & ML models.             |
| **10** | _Feather / Arrow hand-off_             | Zero-copy IPC between stages in separate procs.                         | Pipeline orchestrator, not individual stages.   | Beyond scope of a single mix-in.                  |
| **11** | _Parallel metric evaluation_           | Use `parallel_map` inside Stage 2 loops.                                | `Stage2Imputer` (“per-column metrics”).         | Straightforward edit in that class.               |
| **12** | _Incremental covariance update_        | Rank-1 update instead of re-computing Σ for each candidate.             | `Stage2Imputer._evaluate_impute_num`            | Algebra change, not infra.                        |
| **13** | _Random projection before Mahalanobis_ | Reduce **p** to ≤ 300 dims before Mahalanobis.                          | `OutlierDetector` multivariate section.         | Adds sklearn transform & inverse map.             |
| **14** | _Two-pass winsorisation_               | Light clip first, rerun rules.                                          | `OutlierDetector` fit routine.                  | Stage-specific heuristic.                         |
| **15** | _64-row pre-scan_                      | Decide “simple vs enhanced” on a small sample.                          | `Stage4Transform.fit`                           | Minor logic gate.                                 |
| **16** | _Pickle5 model snapshots_              | Persist column transformers for fast reload.                            | Stages 2 – 4 `.fit()`                           | Needs local path mgmt & versioning.               |

---

# 📄 README.md — _Integrating perfkit v4_

```markdown
# perfkit v4 ⚙️⏱⚡

Light-weight mix-ins for **parallelism, GPU fast-paths and profiling**.

## 1 · Install / drop-in

Place `perfkit.py` anywhere on your `PYTHONPATH`, e.g.
```

project_root/
└─ utils/
└─ perfkit.py # ← v4 file

````

#### Optional system libs
| Feature | Package | Notes |
|---------|---------|-------|
| CPU parallelism | `joblib`, `psutil` | Already required by v4 |
| GPU fast-path   | `cupy` ≥ 10        | Automatically ignored if absent |

---

## 2 · Quick start (inherit one mix-in)

```python
from perfkit import PerfMixin                 # bundles ⏱ ⚙️ ⚡

class Stage2Imputer(PerfMixin):               # <— just add
    def __init__(self, *a,
                 n_jobs=0.5,                  # 50 % cores (float) or 4 (int)
                 use_gpu=True,                # force GPU, False to disable, None=auto
                 **kw):
        super().__init__(*a, n_jobs=n_jobs, use_gpu=use_gpu, **kw)
        ...
````

_No other boiler-plate required._

---

## 3 · Controlling resources

| What               | Code                        | Env override              | Default     |
| ------------------ | --------------------------- | ------------------------- | ----------- |
| CPU workers        | `n_jobs=8` or `n_jobs=0.75` | `PERF_N_JOBS=0.25`        | 0.5 × cores |
| GPU on/off         | `use_gpu=True/False`        | `PERF_USE_GPU=true/false` | auto-detect |
| Fast profiling off | –                           | `FAST_MODE=1`             | off         |
| RAM guard          | –                           | `MAX_RAM_FRACTION=90`     | 95 %        |

---

## 4 · Using `parallel_map`

```python
def _work(col):
    ...                               # pure function
    return result

results = self.parallel_map(_work, self.numeric_cols,
                            min_tasks=4,             # <4 → serial
                            prefer="auto")           # v4 picks threads vs processes
```

_Serial_ fallback avoids joblib overhead for tiny loops.

---

## 5 · GPU helpers

```python
p_val = self.ks_fast(arr1, arr2)             # GPU KS if enabled
sample = self.rand_choice(arr, size=1000)    # GPU random choice
```

Both silently drop to CPU if CuPy/CUDA unavailable.

---

## 6 · Profiling & cost report

Every public method is timed + memory-tracked.

```python
model = Stage4Transform(...).fit(df)
for row in model.report():      # list of dicts (sorted by runtime)
    print(row)
model.export_csv("stage4_perf.csv")          # aggregate CSV dump
```

Fields: `method`, `calls`, `seconds`, `mem_peak_mb`, `mem_Δ_mb`, `cost`.

---

## 7 · Thread-pool hygiene (optional)

```python
import threadpoolctl, os
threadpoolctl.threadpool_limits(n_threads=1)
os.environ["OMP_NUM_THREADS"] = "1"
```

Prevents BLAS or OpenMP from over-subscribing inside each joblib worker.

---

## 8 · Adapting existing loops (example)

**Before**

```python
for col in self.numeric_cols:
    ks = stats.ks_2samp(a, b).pvalue
    rng = np.random.default_rng(seed)
    x  = rng.choice(arr, size)
```

**After**

```python
def _process(col):
    ks = self.ks_fast(a, b)                             # GPU-aware
    x  = self.rand_choice(arr, size, seed=0)
    return ks, x

out = self.parallel_map(_process, self.numeric_cols)
```

---

## 9 · Inter-stage hand-off (advanced)

Perfkit is infra only; wide-matrix tricks (sparse frames, batched blocks,
random projections…) are listed in the **pending ideas (see table)** and
should be implemented **inside** the relevant Stage classes.

---

### ✅ Done

perfkit v4 already covers ideas **1–4, 17, 18**.

### 🕗 Next

Implement ideas **5–16** (see table above) directly in your
`Stage2Imputer`, `OutlierDetector`, `Stage4Transform`, etc.
They are algorithm-specific and cannot live in a generic mix-in.

Happy crunching 🚀

```

Copy the file, update your stages as shown, and you’re good to go!
```
