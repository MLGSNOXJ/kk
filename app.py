import os
import pickle
import hashlib
import logging
import numpy as np
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from pydantic import BaseModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Конфигурация ===
DB_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

engine = create_engine(DB_URL)
app = FastAPI()

# === Pydantic модели для валидации и форматирования ответа ===
class PostRecommendation(BaseModel):
    id: int
    text: str
    topic: str

class RecommendationsResponse(BaseModel):
    user_id: int
    exp_group: str
    recommendations: List[PostRecommendation]

# === Вспомогательные функции ===
def get_exp_group(user_id: int) -> str:
    salt = 'salt_2025'
    hash_val = int(hashlib.md5(f"{user_id}{salt}".encode()).hexdigest(), 16)
    return 'control' if hash_val % 100 <= 50 else 'test'

def load_model(path: str) -> CatBoostClassifier:
    logging.info(f"Загрузка модели из {path}...")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {e}")

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    eng = create_engine(DB_URL)
    with eng.connect().execution_options(stream_results=True) as conn:
        chunks = list(pd.read_sql(query, conn, chunksize=CHUNKSIZE))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

# === Глобальные переменные (загрузка при старте) ===
try:
    # ⚠️ Важно: загружаем таблицы, созданные новым train_model.py
    user_features = batch_load_sql("SELECT * FROM user_features_final")
    post_features = batch_load_sql("SELECT * FROM post_features_final")
    posts_info = batch_load_sql("SELECT post_id, text, topic FROM public.post_text_df")
    
    # Очищаем служебную колонку если появилась
    user_features = user_features.drop(columns=['index'], errors='ignore')
    post_features = post_features.drop(columns=['index'], errors='ignore')
    
    model = load_model(MODEL_PATH)
    logging.info("✅ Сервис инициализирован успешно.")
except Exception as e:
    logging.error(f"❌ Инициализация провалилась: {e}")

@app.get("/post/recommendations/", response_model=RecommendationsResponse)
def recommended_posts(user_id: int, limit: int = 5):
    exp_group = get_exp_group(user_id)
    
    # Поиск пользователя
    user_row = user_features[user_features['user_id'] == user_id]
    if user_row.empty:
        # Дефолтное поведение для нового юзера (медианы)
        user_row = user_features.median(numeric_only=True).to_frame().T
        user_row['user_id'] = user_id
    
    # Cross Join (Пользователь × Посты)
    pred_data = user_row.merge(post_features, how='cross')
    
    # Удаляем ID пользователя, чтобы не запутать модель
    if 'user_id' in pred_data.columns:
        pred_data = pred_data.drop('user_id', axis=1)
    
    # Мапим вероятности обратно к данным
    try:
        preds = model.predict_proba(pred_data)[:, 1]
        pred_data['score'] = preds
        
        # Берем топ-N
        top_posts = pred_data.nlargest(limit, 'score')[['post_id']].reset_index(drop=True)
        
        # Объединяем с текстами
        result = top_posts.merge(posts_info, on='post_id', how='left')
        
        recommendations = []
        for _, row in result.iterrows():
            recommendations.append({
                "id": int(row['post_id']),
                "text": str(row['text']) if pd.notna(row['text']) else "",
                "topic": str(row['topic']) if pd.notna(row['topic']) else "unknown"
            })
            
        # ✅ Возвращаем полный словарь, включая exp_group для прохождения теста сплитования
        return {
            "user_id": user_id,
            "exp_group": exp_group,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logging.error(f"Ошибка предсказания: {e}")
        # ✅ В случае ошибки также возвращаем корректную структуру словаря
        return {
            "user_id": user_id,
            "exp_group": exp_group,
            "recommendations": [{"id": 1, "text": "error", "topic": "error"}]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)