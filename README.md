# ML Final Project — Personalised Recommendation Service

## Project structure

```
ml_final_project/
├── requirements.txt      # Dependencies
├── database.py           # PostgreSQL connection helper
├── schema.py             # Pydantic response models
├── load_data.py          # Step 1: Download raw data from DB → data/
├── train.py              # Steps 2–4: Feature engineering + training → models/
├── app.py                # Step 5–6: FastAPI service
└── README.md
```

---

## Quick start

### 1. Create virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data from PostgreSQL

```bash
mkdir -p data
python load_data.py
```

This saves `data/users.parquet`, `data/posts.parquet`, `data/feed.parquet`.

### 3. Train the model

```bash
mkdir -p models
python train.py
```

Outputs:
- `models/catboost_model.cbm` — trained CatBoost model
- `models/tfidf.pkl` — fitted TF-IDF vectoriser
- `models/svd.pkl` — fitted TruncatedSVD (text → 20 dims)
- `models/post_features.parquet` — pre-computed post feature table
- `models/user_features.parquet` — pre-computed user feature table
- `models/feature_cols.pkl` — ordered list of feature column names

Training prints ROC-AUC on the held-out temporal test split.
A value above **0.62** is considered good for this dataset.

### 4. Start the service

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Test the endpoint

```bash
curl "http://localhost:8000/post/recommendations/?id=200&time=2021-11-25T10:00:00&limit=5"
```

Expected response shape:
```json
{
  "exp_group": "3",
  "recommendations": [
    {"id": 123, "text": "...", "topic": "python"},
    ...
  ]
}
```

---

## Feature overview

| Group | Features |
|---|---|
| User profile | age, gender, country, city, os, source, exp_group |
| User activity | user_views, user_likes, user_like_rate |
| Temporal | hour, dayofweek, month, is_weekend |
| Post stats | post_views, post_likes, post_like_rate |
| Post metadata | topic, text_len, word_count, unique_words |
| Post text (TF-IDF+SVD) | text_svd_0 … text_svd_19 |

## Model

**CatBoostClassifier** with 500 trees, depth 6, learning rate 0.05.
Trained on 80% of data (temporal split), evaluated on most-recent 20%.
Optimisation metric: AUC.

## Evaluation metric (checker system)

The checker uses **hitrate@5**: for each user it checks whether at least one
actually-liked post appears in the top-5 recommendations.

    hitrate@5 = mean over users of 1{liked post ∈ top-5 recommendations}
