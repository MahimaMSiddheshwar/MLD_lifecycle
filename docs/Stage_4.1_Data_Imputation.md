# 📦 Imputation Strategies for Classical Machine Learning

Missing data is ubiquitous in real-world datasets. Proper handling of missing values is critical—poor choices can introduce bias, distort distributions, and degrade model performance. This document provides an **extensive** taxonomy of imputation approaches, guidelines on when to use each, pitfalls to avoid, and code snippets for implementation.

---

## Table of Contents

1. [Missingness Mechanisms](#1-missingness-mechanisms)
   1.1. [Missing Completely at Random (MCAR)](#11-missing-completely-at-random-mcar)
   1.2. [Missing at Random (MAR)](#12-missing-at-random-mar)
   1.3. [Missing Not at Random (MNAR)](#13-missing-not-at-random-mnar)

2. [Drop vs. Fill](#2-drop-vs-fill)
   2.1. [Complete Case Analysis (CCA)](#21-complete-case-analysis-cca)
   2.2. [When to Drop Rows](#22-when-to-drop-rows)

3. [Univariate Imputation](#3-univariate-imputation)
   3.1. [Numeric Features](#31-numeric-features)
    • Mean Imputation
    • Median Imputation
    • Constant / Arbitrary Value Imputation
   3.2. [Categorical Features](#32-categorical-features)
    • Mode (Most Frequent) Imputation
    • Constant / “Missing” Category
    • Random Sample Imputation

4. [Multivariate Imputation](#4-multivariate-imputation)
   4.1. [K-Nearest Neighbors Imputer (KNNImputer)](#41-k-nearest-neighbors-imputer-knnimputer)
   4.2. [Iterative Imputer (MICE)](#42-iterative-imputer-mice)
   4.3. [Pros & Cons of Multivariate Imputation](#43-pros--cons-of-multivariate-imputation)

5. [Random Sample Imputation](#5-random-sample-imputation)
   5.1. [How It Works](#51-how-it-works)
   5.2. [When to Use](#52-when-to-use)
   5.3. [Advantages & Disadvantages](#53-advantages--disadvantages)

6. [Missing Indicator Strategy](#6-missing-indicator-strategy)
   6.1. [Implementation](#61-implementation)
   6.2. [When to Use](#62-when-to-use)
   6.3. [Pitfalls / Tips](#63-pitfalls--tips)

7. [Guidelines & Thresholds](#7-guidelines--thresholds)
   7.1. [Percentage Missing Thresholds](#71-percentage-missing-thresholds)
   7.2. [Assessing MCAR / MAR / MNAR](#72-assessing-mcar--mar--mnar)

8. [Effects on Distribution & Covariance](#8-effects-on-distribution--covariance)

9. [Implementation Examples](#9-implementation-examples)

10. [Summary Checklist & Best Practices](#10-summary-checklist--best-practices)

---

## 1 — Missingness Mechanisms<a name="1-missingness-mechanisms"></a>

Understanding _why_ data is missing is crucial for choosing the right imputation strategy.

### 1.1 Missing Completely at Random (MCAR)<a name="11-missing-completely-at-random-mcar"></a>

- **Definition**: The probability that a value is missing is **independent** of both observed and unobserved data.

  - e.g., A lab technician accidentally spills a sample, so that measurement is lost.

- **Implication**: CCA (dropping missing rows) does not introduce bias (but reduces sample size).

### 1.2 Missing at Random (MAR)<a name="12-missing-at-random-mar"></a>

- **Definition**: The probability of missingness depends only on **observed** data, not on the missing value itself.

  - e.g., Income is more likely to be unreported for younger respondents (age is known), but among the same age group, missingness does not depend on income.

- **Implication**: Imputation methods that condition on observed variables can recover unbiased estimates if the model for missingness is correctly specified (e.g., multivariate imputation).

### 1.3 Missing Not at Random (MNAR)<a name="13-missing-not-at-random-mnar"></a>

- **Definition**: Missingness depends on the **unobserved** value.

  - e.g., People with extremely high income choose not to disclose it.

- **Implication**: Standard imputation can introduce bias. Specialized methods (e.g., modeling missingness mechanism) are needed—beyond scope of classical ML.

---

## 2 — Drop vs. Fill<a name="2-drop-vs-fill"></a>

Two high-level approaches:

1. **Drop rows** (Complete Case Analysis, CCA)
2. **Fill (Impute) missing values**

### 2.1 Complete Case Analysis (CCA)<a name="21-complete-case-analysis-cca"></a>

- **What it is**: Remove any row that contains a missing value in _any_ feature being used.
- **When to Use**:

  - If **MCAR** can be reasonably assumed.
  - If the fraction of missing rows is very low (e.g., < 5 %) so that dropping them does not significantly reduce statistical power.
  - If downstream models cannot handle missing values at all and imputation is undesirable.

- **Pitfalls / Tips**:

  - If missingness is not MCAR, CCA will **bias** estimates.
  - Even if MCAR, dropping too many rows can weaken model training (reduced sample size).

### 2.2 When to Drop Rows<a name="22-when-to-drop-rows"></a>

- **General Rule of Thumb**:

  - If a feature/row has > 50 % missing, consider dropping that **feature** or **row** rather than imputing.
  - If overall rows with any missing value < 5 %, CCA might be acceptable.

- **Feature-Level vs. Row-Level**:

  - If a _column_ has > 50 % missing → drop the column (rarely salvageable).
  - If a _row_ has missing in > N columns (e.g., > 30 % of features) → drop the row.

---

## 3 — Univariate Imputation<a name="3-univariate-imputation"></a>

Impute each feature individually, ignoring relationships with other variables.

### 3.1 Numeric Features<a name="31-numeric-features"></a>

#### 3.1.1 Mean Imputation

- **What it does**: Replace missing values with the **mean** of the non-missing entries in that feature.
- **When to use**:

  - Feature distribution is roughly symmetric (no heavy skew).
  - Missing fraction is small (< 10 %).

- **Implementation**:

  ```python
  from sklearn.impute import SimpleImputer
  import numpy as np

  imputer = SimpleImputer(strategy="mean")
  df["age_imputed"] = imputer.fit_transform(df[["age"]])
  ```

- **Pitfalls / Tips**:

  - **Distorts distribution** by pulling all missing to the mean → underestimates variance.
  - **Mutiplies correlation** artificially: replacing missing with mean reduces covariance with other features.
  - Safe only when missingness is small and truly random.

#### 3.1.2 Median Imputation

- **What it does**: Replace missing values with the **median** (50th percentile) of non-missing entries.
- **When to use**:

  - Distribution is skewed or contains outliers (median is robust).

- **Implementation**:

  ```python
  imputer = SimpleImputer(strategy="median")
  df["income_imputed"] = imputer.fit_transform(df[["income"]])
  ```

- **Pitfalls / Tips**:

  - Still “flattens” distribution at the imputed value (less distortion than mean).
  - Use median if the feature has a long tail.

#### 3.1.3 Constant / Arbitrary Value Imputation

- **What it does**: Fill missing numeric values with a fixed constant (e.g., 0, -1, or a domain-specific sentinel).
- **When to use**:

  - Missingness may encode information itself (e.g., “not applicable” = 0).
  - You want to preserve missing-intactness as a special category.

- **Implementation**:

  ```python
  imputer = SimpleImputer(strategy="constant", fill_value=-1)
  df["score_imputed"] = imputer.fit_transform(df[["score"]])
  ```

- **Pitfalls / Tips**:

  - Fills all missing with the same value → can create artificial clusters.
  - If your model treats -1 as a valid numeric value, distort relationships. Better add a missing indicator (see [Section 6](#6-missing-indicator-strategy)).

### 3.2 Categorical Features<a name="32-categorical-features"></a>

#### 3.2.1 Mode (Most Frequent) Imputation

- **What it does**: Replace missing with the **most frequent** category.
- **When to use**:

  - Category distribution is not too skewed; most frequent indeed “typical.”

- **Implementation**:

  ```python
  from sklearn.impute import SimpleImputer

  imputer = SimpleImputer(strategy="most_frequent")
  df["city_imputed"] = imputer.fit_transform(df[["city"]])
  ```

- **Pitfalls / Tips**:

  - Over-represents the mode category → less variation.
  - If missingness is > 20 %, consider alternate approaches (random sample or supervised).

#### 3.2.2 Constant / “Missing” Category

- **What it does**: Fill missing categorical values with a new category label, e.g., `"__MISSING__"`.
- **When to use**:

  - Missingness itself may carry metadata (e.g., user refused to answer).
  - You want to preserve “missing” as a distinct category.

- **Implementation**:

  ```python
  df["color_imputed"] = df["color"].fillna("__MISSING__")
  ```

- **Pitfalls / Tips**:

  - If `"__MISSING__"` ends up being very frequent, it can distort encoding.
  - For one-hot encoding, this creates an extra column—ensure model can handle it.

#### 3.2.3 Random Sample Imputation for Categoricals

- **What it does**: Replaces each missing category with a random draw from the **observed distribution** of that feature.
- **When to use**:

  - You believe missing are MAR and want to preserve original category distribution.
  - When missingness is low-to-moderate (< 10 %) and categories fairly balanced.

- **Implementation**:

  ```python
  import numpy as np

  def random_cat_impute(series: pd.Series) -> pd.Series:
      non_null = series.dropna().values
      return series.apply(lambda x: np.random.choice(non_null) if pd.isna(x) else x)

  df["payment_type_imputed"] = random_cat_impute(df["payment_type"])
  ```

- **Pitfalls / Tips**:

  - Introduces randomness → results vary per seed.
  - Preserves marginal distribution but ignores covariate relationships.

---

## 4 — Multivariate Imputation<a name="4-multivariate-imputation"></a>

Leverage other features to impute missing values—especially useful under **MAR**.

### 4.1 K-Nearest Neighbors Imputer (KNNImputer)<a name="41-k-nearest-neighbors-imputer-knnimputer"></a>

- **What it does**: For a row with missing entries, find its _k_ nearest neighbors (based on distance in the feature space of observed values) and impute missing with the average (numeric) or most frequent (categorical via extension) among neighbors.
- **When to use**:

  - Numeric features with correlated counterparts (e.g., height and weight).
  - Better than univariate if relationships are strong.

- **Implementation (scikit-learn)**:

  ```python
  from sklearn.impute import KNNImputer

  knn_imp = KNNImputer(n_neighbors=5, weights="uniform")
  numeric_cols = df.select_dtypes(include="number").columns
  df[numeric_cols] = knn_imp.fit_transform(df[numeric_cols])
  ```

- **Pitfalls / Tips**:

  - **Compute cost** scales with (n_rows × n_columns) to find neighbors—slow for large data.
  - Requires **scaling** first (e.g., StandardScaler) so that distance metric is meaningful.
  - Does not natively handle categorical—either encode categories first or use separate strategies.

### 4.2 Iterative Imputer (MICE)<a name="42-iterative-imputer-mice"></a>

- **What it does**: Also known as _Multiple Imputation by Chained Equations (MICE)_. Iteratively models each feature with missing values as a function of other features (regression/classification) and imputes.

  - At each iteration, one feature’s missing values are predicted using a regression model (e.g., BayesianRidge for continuous, RandomForest for categorical).
  - Iterate until convergence or max iterations.

- **When to use**:

  - Complex relationships among features.
  - Better at preserving multivariate distributions.

- **Implementation (scikit-learn)**:

  ```python
  from sklearn.experimental import enable_iterative_imputer  # noqa
  from sklearn.impute import IterativeImputer
  from sklearn.linear_model import BayesianRidge

  imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
  numeric_cols = df.select_dtypes(include="number").columns
  df[numeric_cols] = imp.fit_transform(df[numeric_cols])
  ```

- **Pitfalls / Tips**:

  - Slower than KNN; may take time for large datasets.
  - Can propagate errors if model assumptions are violated.
  - Does not natively support categorical—encode categories first or use specialized wrappers.

### 4.3 Pros & Cons of Multivariate Imputation<a name="43-pros--cons-of-multivariate-imputation"></a>

| Method               | Pros                                                                                                                                     | Cons                                                                                 |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **KNNImputer**       | Simple to implement; preserves local structure; no model assumptions                                                                     | Slow on large data; sensitive to scaling; doesn’t handle categorical natively        |
| **IterativeImputer** | Captures complex relationships; preserves multivariate distributions; can handle both numeric & categorical (with appropriate estimator) | Computationally expensive; might not converge; requires careful tuning of regressors |

---

## 5 — Random Sample Imputation<a name="5-random-sample-imputation"></a>

A versatile technique applicable to **both** numeric and categorical features.

### 5.1 How It Works<a name="51-how-it-works"></a>

- **Numeric Version**: For each missing value in feature _X_, randomly sample a non-missing value from _X_’s observed distribution and use it.
- **Categorical Version**: Similarly, draw a category at random from the set of observed categories (with or without weighting by frequency).

### 5.2 When to Use<a name="52-when-to-use"></a>

- Feature is **MCAR** (no relationship between missingness and other variables).
- You want to **preserve marginal distribution** (unlike mean/median which collapse missing to a single value).
- Missing fraction is **low-to-moderate** (< 15 %).

### 5.3 Advantages & Disadvantages<a name="53-advantages--disadvantages"></a>

| Aspect                 | Advantages                                                                         | Disadvantages                                                                                                                             |
| ---------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **Preserves Variance** | Random sampling maintains the original variability of that feature                 | Introduces sampling noise → multiple runs yield different imputations                                                                     |
| **Simplicity**         | Very simple to implement                                                           | Does not account for feature correlations (multivariate relationships lost)                                                               |
| **Linear Models**      | Well-suited; does not bias mean/median-based estimates                             | May slightly degrade performance for tree-based models if missingness patterns are complex                                                |
| **Storage**            | Need to store the “observed distribution” (list of values) in memory for inference | If distribution changes between train and test (new categories or numeric ranges), random sampling from train distribution may be invalid |
| **Outlier Insertion**  | Low risk of creating outliers (samples from real values)                           | If missing share is high, repeated sampling may over-represent outliers                                                                   |

---

## 6 — Missing Indicator Strategy<a name="6-missing-indicator-strategy"></a>

Rather than discarding the fact of missingness, explicitly encode it as a separate binary feature.

### 6.1 Implementation<a name="61-implementation"></a>

```python
# Suppose df["income"] has missing values
df["income_missing_flag"] = df["income"].isna().astype(int)

# Then impute "income" separately (e.g., mean or median)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
df["income_imputed"] = imp.fit_transform(df[["income"]])
```

### 6.2 When to Use<a name="62-when-to-use"></a>

- Missingness itself **may carry information** (MAR or MNAR).
- Certain algorithms (e.g., tree-based) can exploit missing flags to split rules.

### 6.3 Pitfalls / Tips<a name="63-pitfalls--tips"></a>

- If missingness is truly MCAR, indicator may add noise without benefit.
- Avoid including the original “missing” placeholder value (e.g., `-1`) along with a missing flag; use imputer + missing flag instead of constant imputation alone.

---

## 7 — Guidelines & Thresholds<a name="7-guidelines--thresholds"></a>

### 7.1 Percentage Missing Thresholds<a name="71-percentage-missing-thresholds"></a>

- **< 5 % missing (per feature)**:
  • CCA (drop rows) is often safe if MCAR.
  • Univariate imputation (mean/median/mode) introduces acceptable distortion.

- **5 %–20 % missing**:
  • Avoid CCA if non-MCAR;
  • Prefer imputation (univariate or random sample) combined with missing indicator.

- **> 20 %–50 % missing**:
  • Consider multivariate imputation (KNN, MICE) or evaluate whether feature should be dropped.
  • If feature is critical, multivariate methods can salvage it.

- **> 50 % missing**:
  • Strongly consider dropping the feature unless domain reasons suggest imputation is warranted.

### 7.2 Assessing MCAR / MAR / MNAR<a name="72-assessing-mcar--mar--mnar"></a>

- **Little’s MCAR Test** (statistical test) can help detect MCAR vs. MAR/MNAR (→ `statsmodels` or specialized packages).
- **Visual Inspection**:

  - Plot missingness vs. other features (e.g., missing income vs. age).
  - If missingness correlates with observed features → MAR likely.

- **Domain Knowledge**:

  - If missingness is due to user actions (e.g., people “don’t want to reveal income” if income is high) → MNAR.

---

## 8 — Effects on Distribution & Covariance<a name="8-effects-on-distribution--covariance"></a>

| Method                       | Impact on Distribution                                                                   | Impact on Covariance / Correlation                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Mean Imputation**          | Collapses all missing to mean → underestimates variance; distribution becomes “spikier”  | Inflates correlation with other features (imputed values identical)                            |
| **Median Imputation**        | Collapses missing to median → less distortion for skewed distributions                   | Similar to mean: reduces variability, can artificially tighten relationships                   |
| **Constant Mode Imputation** | Creates a spike at the mode → underestimates spread                                      | If mode category correlates with target, can bias model                                        |
| **Random Sample Imputation** | Preserves marginal distribution of feature                                               | Maintains original feature correlation structure (on average); some sampling noise introduced  |
| **KNN Imputation**           | Preserves local structure; distribution closer to true                                   | Preserves multivariate correlations if k is chosen well                                        |
| **Iterative Imputation**     | Aims to preserve multivariate distribution; distribution depends on modeling assumptions | Can maintain covariance structure if models are correctly specified; may introduce overfitting |

---

## 9 — Implementation Examples<a name="9-implementation-examples"></a>

Below are illustrative code snippets for common imputation approaches. Adapt column lists and parameters as needed.

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Sample DataFrame
df = pd.DataFrame({
    "age": [25, np.nan, 45, 30, np.nan, 55],
    "income": [50_000, 60_000, np.nan, 80_000, 90_000, np.nan],
    "city": ["NY", "LA", np.nan, "SF", "LA", "NY"],
    "rating": [4, 5, np.nan, 3, np.nan, 2]
})

# -----------------------------
# 1. DROP vs. FILL
# -----------------------------
# 1a. CCA: drop any row with missing values
df_cca = df.dropna(subset=["age","income","city","rating"])

# -----------------------------
# 2. UNIVARIATE IMPUTATION
# -----------------------------
# Numeric: Mean / Median / Constant
mean_imp = SimpleImputer(strategy="mean")
df["age_mean_imputed"] = mean_imp.fit_transform(df[["age"]])

median_imp = SimpleImputer(strategy="median")
df["income_med_imputed"] = median_imp.fit_transform(df[["income"]])

const_imp = SimpleImputer(strategy="constant", fill_value=-1)
df["rating_const_imputed"] = const_imp.fit_transform(df[["rating"]])

# Categorical: Mode / Constant / Random Sample
mode_imp = SimpleImputer(strategy="most_frequent")
df["city_mode_imputed"] = mode_imp.fit_transform(df[["city"]])

df["city_const_imputed"] = df["city"].fillna("__UNKNOWN__")

def random_cat_impute(series):
    observed = series.dropna().values
    return series.apply(lambda x: np.random.choice(observed) if pd.isna(x) else x)

df["city_random_imputed"] = random_cat_impute(df["city"])

# -----------------------------
# 3. MULTIVARIATE IMPUTATION
# -----------------------------
# 3a. KNNImputer
knn_imp = KNNImputer(n_neighbors=3)
numeric_cols = ["age", "income", "rating"]
df[numeric_cols] = knn_imp.fit_transform(df[numeric_cols])

# 3b. IterativeImputer (MICE)
mice_imp = IterativeImputer(max_iter=10, random_state=0)
df[numeric_cols] = mice_imp.fit_transform(df[numeric_cols])

# -----------------------------
# 4. RANDOM SAMPLE IMPUTATION (NUMERIC)
# -----------------------------
def random_numeric_impute(series):
    observed = series.dropna().values
    return series.apply(lambda x: np.random.choice(observed) if pd.isna(x) else x)

df["income_random_imputed"] = random_numeric_impute(df["income"])

# -----------------------------
# 5. MISSING-INDICATOR STRATEGY
# -----------------------------
df["age_missing_flag"] = df["age"].isna().astype(int)
df["age_imputed_for_flag"] = SimpleImputer(strategy="median").fit_transform(df[["age"]])

# -----------------------------
# 6. COMPLETE CASE ANALYSIS (CCA) FOR SPECIFIC COLUMNS
# -----------------------------
# Only drop rows if missing in both age & income (e.g., not severe)
df_filtered = df.dropna(subset=["age", "income"], how="all")

# -----------------------------
# 7. BINNING (Example)
# -----------------------------
# Unsupervised: Quantile bins for income
from sklearn.preprocessing import KBinsDiscretizer
qf = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
df["income_qbin"] = qf.fit_transform(df[["income"]])

# Supervised: Decision tree-based binning (example for 'age' vs 'rating' as proxy)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
dt.fit(df[["age"]].fillna(df["age"].median()), (df["rating"] > df["rating"].median()).astype(int))
thresholds = sorted(set(dt.tree_.threshold[dt.tree_.threshold != -2]))
df["age_dtbin"] = pd.cut(df["age"], bins=[-np.inf]+thresholds+[np.inf], labels=False)
```

---

## 10 — Summary Checklist & Best Practices<a name="10-summary-checklist--best-practices"></a>

1. **Assess Missingness Mechanism**

   - Perform Little’s MCAR test or visual checks to see if missingness correlates with other features.
   - If MCAR and missing < 5 %, CCA is acceptable.

2. **Drop vs. Impute Decision**

   - If a feature has > 50 % missing, consider dropping the feature.
   - If a row has missing in > 30 % of features, dropping the row may be preferable.

3. **Univariate Imputation**

   - **Numeric**:

     - If distribution symmetric → **mean**.
     - If skewed/outliers → **median**.
     - If missingness encodes meaning → **constant** + missing indicator.

   - **Categorical**:

     - **Mode** when missing < 10 % & category distribution balanced.
     - **Constant (“**MISSING**”)** if missingness itself is informative.
     - **Random sample** to preserve marginal distribution when missing < 15 %.

4. **Multivariate Imputation**

   - Use **KNNImputer** when numeric features are low-dimensional (< 50) and relationships are local.
   - Use **IterativeImputer (MICE)** when capturing complex multivariate relationships and missingness is MAR.

5. **Random Sample Imputation**

   - Ideal when you want to preserve variance and missingness is MCAR.
   - Be aware of added randomness—set a seed for reproducibility.

6. **Missing Indicator**

   - Add a binary flag whenever imputation may hide informative missingness (MAR or MNAR).
   - Many tree-based models can utilize missing flags to improve splits.

7. **Evaluate Imputation Impact**

   - After imputation, compare **feature distributions** (histogram, boxplot) before vs. after.
   - Check whether **covariances/correlations** differ significantly.

8. **Document Thoroughly**

   - For each feature, record original missing percentage and imputation approach in a **Feature Dictionary**.
   - Note whether missingness was MCAR/MAR/MNAR and justify the chosen strategy.

9. **Avoid Data Leakage**

   - Always fit imputer (especially multivariate or supervised) on **training set only**.
   - Serialize (pickle) imputer parameters and apply to validation/test.

10. **Automate & Validate**

    - Wrap imputation into a `Pipeline` or `ColumnTransformer` for consistent application.
    - Include unit-tests for custom imputation functions (e.g., random sampling) to ensure stable behavior.

---

> ⚠️ **Key Takeaway**: There is **no one-size-fits-all**.
>
> - If missingness is small (< 5 %) and MCAR → CCA or simple univariate imputation.
> - If relationships exist among features (MAR), use multivariate methods (KNN, MICE).
> - Always check the distributional impact and document every step.

Use this guide to ensure your imputation choices are principled, reproducible, and appropriate for your dataset.
