"""
Steps 2–4: Feature engineering, model training, saving artefacts.

Run after load_data.py has populated the data/ folder.
Outputs saved to models/ folder.
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# 1. Load raw data
# ══════════════════════════════════════════════════════════════════════════
print("Loading parquet files…")
users_df = pd.read_parquet("data/users.parquet")
posts_df = pd.read_parquet("data/posts.parquet")
feed_df  = pd.read_parquet("data/feed.parquet")

print(f"users {users_df.shape}, posts {posts_df.shape}, feed {feed_df.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Filter feed to VIEW events only (target is defined there)
# ══════════════════════════════════════════════════════════════════════════
views = feed_df[feed_df["action"] == "view"].copy()
views["timestamp"] = pd.to_datetime(views["timestamp"])

# ══════════════════════════════════════════════════════════════════════════
# 3. Temporal features
# ══════════════════════════════════════════════════════════════════════════
views["hour"]       = views["timestamp"].dt.hour
views["dayofweek"]  = views["timestamp"].dt.dayofweek
views["month"]      = views["timestamp"].dt.month
views["is_weekend"] = (views["dayofweek"] >= 5).astype(int)

# ══════════════════════════════════════════════════════════════════════════
# 4. Post features: TF-IDF → SVD (20 dims) + topic + text stats
# ══════════════════════════════════════════════════════════════════════════
print("Building post TF-IDF features…")
tfidf = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(posts_df["text"].fillna(""))

svd = TruncatedSVD(n_components=20, random_state=42)
tfidf_svd = svd.fit_transform(tfidf_matrix)

post_svd_df = pd.DataFrame(
    tfidf_svd,
    columns=[f"text_svd_{i}" for i in range(20)],
)
post_svd_df["post_id"] = posts_df["id"].values

# Text stats
posts_df["text_len"]      = posts_df["text"].str.len()
posts_df["word_count"]    = posts_df["text"].str.split().str.len()
posts_df["unique_words"]  = posts_df["text"].str.split().apply(lambda x: len(set(x)) if isinstance(x, list) else 0)

posts_features = posts_df[["id", "topic", "text_len", "word_count", "unique_words"]].copy()
posts_features = posts_features.rename(columns={"id": "post_id"})
posts_features = posts_features.merge(post_svd_df, on="post_id")

# ══════════════════════════════════════════════════════════════════════════
# 5. Post popularity (like-rate from feed)
# ══════════════════════════════════════════════════════════════════════════
post_stats = (
    views.groupby("post_id")
    .agg(post_views=("target", "count"), post_likes=("target", "sum"))
    .reset_index()
)
post_stats["post_like_rate"] = post_stats["post_likes"] / post_stats["post_views"].clip(lower=1)
posts_features = posts_features.merge(post_stats, on="post_id", how="left")
posts_features[["post_views", "post_likes", "post_like_rate"]] = (
    posts_features[["post_views", "post_likes", "post_like_rate"]].fillna(0)
)

# ══════════════════════════════════════════════════════════════════════════
# 6. User features from profile
# ══════════════════════════════════════════════════════════════════════════
user_features = users_df[["user_id", "age", "gender", "country", "city",
                           "exp_group", "os", "source"]].copy()

# User activity stats
user_stats = (
    views.groupby("user_id")
    .agg(user_views=("target", "count"), user_likes=("target", "sum"))
    .reset_index()
)
user_stats["user_like_rate"] = user_stats["user_likes"] / user_stats["user_views"].clip(lower=1)
user_features = user_features.merge(user_stats, on="user_id", how="left")
user_features[["user_views", "user_likes", "user_like_rate"]] = (
    user_features[["user_views", "user_likes", "user_like_rate"]].fillna(0)
)

# ══════════════════════════════════════════════════════════════════════════
# 7. Merge everything into training table
# ══════════════════════════════════════════════════════════════════════════
print("Merging features…")
df = (
    views[["user_id", "post_id", "hour", "dayofweek", "month", "is_weekend", "target"]]
    .merge(user_features, on="user_id", how="left")
    .merge(posts_features, on="post_id", how="left")
)
print(f"Training table shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════
# 8. Train/test split — temporal: last 20% of time is test
# ══════════════════════════════════════════════════════════════════════════
# Sort by original timestamp (re-merge it)
ts = views[["user_id", "post_id", "timestamp"]].copy()
df = df.merge(ts.drop_duplicates(["user_id", "post_id"]), on=["user_id", "post_id"], how="left")

cutoff = df["timestamp"].quantile(0.8)
train_df = df[df["timestamp"] <= cutoff].copy()
test_df  = df[df["timestamp"]  > cutoff].copy()
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

cat_cols = ["gender", "country", "city", "exp_group", "os", "source", "topic"]
drop_cols = ["user_id", "post_id", "timestamp", "target"]

X_train = train_df.drop(columns=drop_cols)
y_train = train_df["target"]
X_test  = test_df.drop(columns=drop_cols)
y_test  = test_df["target"]

# Fill NA in categoricals
for col in cat_cols:
    X_train[col] = X_train[col].fillna("unknown").astype(str)
    X_test[col]  = X_test[col].fillna("unknown").astype(str)

# ══════════════════════════════════════════════════════════════════════════
# 9. Train CatBoost
# ══════════════════════════════════════════════════════════════════════════
print("Training CatBoostClassifier…")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_cols,
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
)
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True,
)

# ══════════════════════════════════════════════════════════════════════════
# 10. Evaluate
# ══════════════════════════════════════════════════════════════════════════
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"\n=== ROC-AUC on test: {auc:.4f} ===\n")

# ══════════════════════════════════════════════════════════════════════════
# 11. Save artefacts
# ══════════════════════════════════════════════════════════════════════════
print("Saving artefacts…")
model.save_model("models/catboost_model.cbm")

with open("models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("models/svd.pkl", "wb") as f:
    pickle.dump(svd, f)

# Save post features table (needed at inference time)
posts_features.to_parquet("models/post_features.parquet", index=False)

# Save user features table (needed at inference time)
user_features.to_parquet("models/user_features.parquet", index=False)

# Save feature column order (without target / ids / timestamp)
feature_cols = [c for c in X_train.columns]
with open("models/feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("Done! Artefacts saved to models/")
print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]} …")
