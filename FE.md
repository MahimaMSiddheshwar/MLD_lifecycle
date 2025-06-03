**Data Cleaning** and **Feature Transformation** both live in the preprocessing stage of an ML pipeline, but they serve distinct purposes:

---

### 📋 Data Cleaning (Phase 3)

**Goal:** Make your raw data “correct and consistent” so that downstream algorithms won’t choke on invalid inputs.

Typical tasks include:

1. **Handling Missing Values**

   - _Identify_ which columns/rows have gaps.
   - _Decide_ to drop, impute (median/mode/KNN/MICE), or flag them.
   - _Example_: If “age” is null for 2 % of records, you might impute with median.

2. **Deduplication & Invariants**

   - _Remove_ exact‐duplicate rows.
   - _Drop_ columns with zero or near‐zero variance (e.g. a column that is “TRUE” for 99.9 % of rows).
   - _Example_: If “user_id” appears twice with identical data, keep only one.

3. **Type & Format Enforcement**

   - _Convert_ strings like `"2023-01-01"` into `datetime` objects.
   - _Coerce_ numeric‐looking strings (`"42.0"`) into floats/ints.
   - _Validate_ ranges or categories via a schema (Pandera, Cerberus, etc.).
   - _Example_: Ensure `“salary”` is a float ≥ 0; any negative values get flagged.

4. **Outlier Detection & Treatment**

   - _Flag_ or _remove_ points beyond 3 σ, outside IQR × 1.5, or via Isolation Forest.
   - Decide whether to cap/floor, drop, or leave them for later.
   - _Example_: A transaction amount of \$1 M when the 99th percentile is \$1 000 likely is a typo.

5. **Basic Value Normalization (sometimes overlaps)**

   - _Strip_ leading/trailing whitespace, lowercase all categorical text, remove non‑ASCII.
   - _Standardize_ date formats (e.g. “MM/DD/YYYY” → ISO).
   - _Example_: Convert “NYC” vs “New York City” into a single canonical category.

6. **Logging & Lineage**

   - _Record_ which rows/columns were dropped or imputed, with timestamps and checksums.
   - _Version_ the cleaned snapshot (e.g. `data/interim/clean.parquet`) so you can reproduce exactly.

_If you imagine “Data Cleaning” as making the dataset technically valid—no NaNs where they shouldn’t be, no nonsensical values, no duplicated keys—then you’re on the right track._

---

### 🔄 Feature Transformation (Phase 5)

**Goal:** Take your “cleaned” data and convert/augment it into representations that are more predictive for your algorithms.

Key activities include:

1. **Scaling & Normalization**

   - _Scale_ numerical columns via StandardScaler, MinMax, RobustScaler, PowerTransformer, or QuantileTransformer.
   - _Log‑transform_ strongly skewed distributions (`np.log1p`).
   - _Example_: Apply `StandardScaler` to “age” and “income” so they both have zero mean / unit variance.

2. **Encoding Categorical Variables**

   - _One‑hot_ for low‑cardinality text columns.
   - _Ordinal_ if there’s a natural order (e.g. “low”, “medium”, “high” → 0, 1, 2).
   - _Target/WOE/Frequency/Hash_ encoding for high‑cardinality categories.
   - _Example_: Use TargetEncoder on “zipcode” (hundreds of unique values) to avoid blowing up feature space.

3. **Binning / Discretization**

   - _Quantile bins_ or _k‑means bins_ (e.g. group “age” into 4 bins).
   - _Custom cutpoints_ if you know domain thresholds (e.g. “income” < 30 k, 30–60 k, > 60 k).
   - _Example_: `KBinsDiscretizer(n_bins=5, strategy='quantile')` to bucket a continuous variable.

4. **Creating Derived or Interaction Features**

   - _Ratios_: e.g. `spend_per_visit = total_spend / num_visits`.
   - _Polynomials_: e.g. adding `age²` or any cross‑term `height × weight`.
   - _Date‑based_: extract year/month/day‐of‐week or compute “days_since_signup”.
   - _Example_: If you have “order_date” and “signup_date”, create `days_since_signup = order_date – signup_date`.

5. **Text Vectorization (light‐weight)**

   - _CountVectorizer_ or _TF‑IDF_ for short text fields (e.g. product reviews).
   - _Text length_ / _word count_ as numeric features if the text is very short.
   - _Example_: Build a TF‑IDF matrix for “review_text” with a cap of 100 tokens.

6. **Dimensionality Reduction (optional within FE)**

   - _PCA_, _UMAP_, or _TruncatedSVD_ to reduce large sparse text/one‑hot matrices.
   - _Example_: Apply `TruncatedSVD(n_components=20)` on TF‑IDF vectors to get 20 components.

7. **Imbalance Adjustment (embedded in FE)**

   - _SMOTE_ or _NearMiss_ applied only on training set after transformations.
   - _(Note: Rarely done as part of FE workflow, but often integrated in the pipeline fit step.)_
   - _Example_: If “fraud” ratio is 1 % positive, run SMOTE to synthetically oversample minority after encoding.

8. **Custom Plug‑ins & Expert Features**

   - Any domain‑specific transform you write (e.g. “extract sentiment from comments”).
   - Unit tests to ensure these transforms always return expected DataFrame shapes.
   - _Example_: A function `add_sentiment_score(df)` that uses a pre‑trained sentiment lexicon to output a new column.

9. **Feature Pruning & Audit**

   - _Near‑zero variance_ filtered before heavy transforms.
   - _Correlation filter_ (drop one of highly correlated pair).
   - _Mutual information filter_ (drop features below bottom 10 % MI).
   - _Feature audits_: record number of features in/out and sparsity.
   - _Example_: Run `VarianceThreshold(1e‑5)` on numeric columns to drop constant or almost constant features.

_In short: Feature Transformation is about “reshaping and enriching” the cleansed data into numerical arrays (or sparse matrices) your model can digest, while optionally pruning irrelevant or redundant features._

---

### 🔑 Where the Line Blurs

- **“Is imputing missing values data cleaning or feature transformation?”**
  ­— It’s **data cleaning** because you’re ensuring no NaNs remain.
- **“Is `np.log1p(column)` cleaning or transforming?”**
  ­— It’s strictly **feature transformation**, since you’re reshaping a valid numeric column for better modeling.
- **“Dropping a column because it has 99.9 % a single value—where does that live?”**
  ­— It can appear in **data cleaning** (as an invariant/dedup step) or in **feature selection** (pruning a useless predictor). In practice, most pipelines treat variance‐based drops as part of “cleaning,” but you can also argue it’s an FE‐pruning step.

---

### 🏗️ Typical Workflow Placement

```text
Phase 3 – Data Cleaning:
  • Deduplicate rows
  • Enforce schemas & dtypes (Pandera)
  • Handle missing (drop / median‐mode / KNN)
  • Detect & treat outliers (IQR / zscore / IF)
  • Normalize text (lowercase, strip whitespace)
  • Save “cleaned” snapshot → data/interim/clean.parquet

Phase 4 – EDA & Feature Selection:
  • Run univariate, bivariate, multivariate analyses
  • Identify leaky or useless features (perfect corr with target)
  • Drop irrelevant columns (IDs, future timestamps)
  • Early split (train/test) if desired to prevent leakage
  • Save “selected” features → data/processed/selected.parquet

Phase 5 – Feature Engineering:
  • Scale & power‐transform numeric columns
  • Encode categorical columns (one‐hot, target, etc.)
  • Bin numeric columns (quantile/kmeans)
  • Create derived features (ratios, date deltas, interactions)
  • Vectorize light text fields (TF‑IDF + text length)
  • Prune low‑MI / highly correlated features
  • Optionally oversample (SMOTE) on train set only
  • Produce final preprocessor → models/preprocessor.joblib
```

---

### 📌 Summary

- **Data Cleaning** = fix “bad” data (missing, invalid, duplicate, outlier).
- **Feature Transformation** = take “good” data and reshape or augment it (scale, encode, derive new columns) to maximize predictive signal.

Keeping these separate ensures you (a) never leak test information into your transforms, and (b) maintain a clear lineage: one snapshot for “cleaned” data, another for “engineered” features.
