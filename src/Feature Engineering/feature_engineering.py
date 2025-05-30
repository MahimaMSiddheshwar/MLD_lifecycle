from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                   OrdinalEncoder, PolynomialFeatures)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from category_encoders.woe import WOEEncoder
import featuretools as ft
import umap

num_cols = ["age", "amount", "ses_length"]
cat_cols = ["city", "plan", "event_channel"]

# ── 4·1 Pipelines
pre = ColumnTransformer([
    ("num", Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ── 4·2 Target/Weight-of-Evidence Encoding
woe = WOEEncoder(cols=["city"])
df_woe = woe.fit_transform(df[["city"]], df["is_churn"])

# ── 4·3 Imbalance
smote = SMOTE(k_neighbors=5, random_state=0)

# ── 4·4 Dimensionality Reduction
pca = PCA(n_components=0.95, random_state=1)     # keeps 95 % variance
u = umap.UMAP(n_neighbors=15, min_dist=0.2)    # manifold for viz

# ── 4·5 Auto-feature generation (FeatureTools)
es = ft.EntitySet().add_dataframe("churn", df,
                                  index="uid", time_index="last_login")
fm, fd = ft.dfs(entityset=es,
                target_dataframe_name="churn",
                max_depth=1,
                verbose=True)
