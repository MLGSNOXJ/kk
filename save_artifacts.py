import pickle
import pandas as pd
import joblib
from database import postgres_connection
from sqlalchemy import create_engine
import warnings
print("🚀 СКРИПТ ЗАПУЩЕН!")  # ← ДОБАВЬТЕ ЭТУ СТРОКУ
warnings.filterwarnings('ignore')

# ==================== 1. СОХРАНЕНИЕ МОДЕЛИ ====================
print("💾 Сохраняем модель...")

# Загружаем обученную модель (если ещё не в памяти)
model = joblib.load("data/model.pkl")

# Проверяем, что модель имеет нужные методы
assert hasattr(model, 'predict_proba'), "❌ Модель должна иметь метод predict_proba()"
assert hasattr(model, 'predict'), "❌ Модель должна иметь метод predict()"

# Сохраняем в pickle (универсальный формат для проверяющей системы)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Модель сохранена в model.pkl")

# ==================== 2. ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ БД ====================
print("\n🔧 Готовим признаки для загрузки в БД...")

# Загружаем исходные данные для реконструкции признаков
# Загружаем исходные данные для реконструкции признаков
df_users = pd.read_parquet("data/users.parquet")
df_posts = pd.read_parquet("data/posts.parquet")
df_feed = pd.read_parquet("data/feed.parquet")

# 🔧 Приводим названия столбцов к ожидаемому формату
if 'post_id' in df_posts.columns and 'id' not in df_posts.columns:
    df_posts.rename(columns={'post_id': 'id'}, inplace=True)
if 'id' in df_users.columns and 'user_id' not in df_users.columns:
    df_users.rename(columns={'id': 'user_id'}, inplace=True)

# Повторяем логику создания признаков (упрощённая версия для inference)
def prepare_user_features_for_inference(df_users: pd.DataFrame, df_feed: pd.DataFrame) -> pd.DataFrame:
    """Готовит признаки пользователей для сохранения в БД."""
    features = df_users.copy()
    
    # Агрегации по истории (только просмотры)
    feed_views = df_feed[df_feed['action'] == 'view'].copy()
    
    user_stats = feed_views.groupby('user_id').agg(
        user_total_views=('post_id', 'count'),
        user_total_likes=('target', 'sum'),
    ).reset_index()
    user_stats['user_ctr'] = user_stats['user_total_likes'] / (user_stats['user_total_views'] + 1)
    
    features = features.merge(user_stats, on='user_id', how='left')
    features['user_total_views'] = features['user_total_views'].fillna(0)
    features['user_total_likes'] = features['user_total_likes'].fillna(0)
    features['user_ctr'] = features['user_ctr'].fillna(0)
    
    # Приводим типы для экономии памяти
    features['user_total_views'] = features['user_total_views'].astype('int32')
    features['user_total_likes'] = features['user_total_likes'].astype('int32')
    features['user_ctr'] = features['user_ctr'].astype('float32')
    
    return features

def prepare_post_features_for_inference(df_posts: pd.DataFrame, df_feed: pd.DataFrame) -> pd.DataFrame:
    """Готовит признаки постов для сохранения в БД."""
    features = df_posts.copy()
    
    # Простые признаки текста
    features['text_length'] = features['text'].astype(str).str.len()
    features['word_count'] = features['text'].astype(str).str.split().str.len()
    
    # Популярность
    feed_likes = df_feed[(df_feed['action'] == 'view') & (df_feed['target'] == 1)].copy()
    post_stats = feed_likes.groupby('post_id').agg(
        post_total_likes=('target', 'count'),
    ).reset_index()
    
    features = features.merge(post_stats, left_on='id', right_on='post_id', how='left')
    features['post_total_likes'] = features['post_total_likes'].fillna(0).astype('int32')
    
    return features

# Создаём датафреймы с признаками
user_features_df = prepare_user_features_for_inference(df_users, df_feed)
post_features_df = prepare_post_features_for_inference(df_posts, df_feed)

print(f"✅ Признаки пользователей: {user_features_df.shape}")
print(f"✅ Признаки постов: {post_features_df.shape}")

# ==================== 3. ЗАГРУЗКА ПРИЗНАКОВ В БД ====================
print("\n📤 Загружаем признаки в PostgreSQL...")

# 🔑 ВАЖНО: замените на ваш уникальный префикс (логин в курсе или случайная строка)
USER_PREFIX = "frisx"  # ← ПОМЕНЯЙТЕ НА СВОЙ!

# Строка подключения (из задания)
# Если у вас другие креды — подставьте свои
DB_URI = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"

engine = create_engine(DB_URI)

try:
    # Сохраняем признаки пользователей
    user_features_df.to_sql(
        f"{USER_PREFIX}_user_features",
        con=engine,
        if_exists="replace",  # перезапишет при повторном запуске
        index=False,
        method="multi",
        chunksize=1000
    )
    print(f"✅ Таблица {USER_PREFIX}_user_features загружена")
    
    # Сохраняем признаки постов
    post_features_df.to_sql(
        f"{USER_PREFIX}_post_features",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000
    )
    print(f"✅ Таблица {USER_PREFIX}_post_features загружена")
    
except Exception as e:
    print(f"⚠️ Ошибка при загрузке в БД: {e}")
    print("💡 Убедитесь, что строка подключения верная и у вас есть права на запись")

# ==================== 4. ФИНАЛЬНАЯ ПРОВЕРКА ====================
print("\n🔍 Финальная проверка модели...")

# Загружаем и тестируем
with open("model.pkl", "rb") as f:
    loaded = pickle.load(f)

# Проверяем методы
print(f"✅ predict: {callable(getattr(loaded, 'predict', None))}")
print(f"✅ predict_proba: {callable(getattr(loaded, 'predict_proba', None))}")

# Быстрый тест на фиктивных данных (если модель ожидает 29 признаков)
import numpy as np
dummy_input = np.zeros((1, 29))  # 29 — количество признаков из вашего X_train
try:
    proba = loaded.predict_proba(dummy_input)
    print(f"✅ Модель работает: predict_proba возвращает {proba.shape}")
except Exception as e:
    print(f"⚠️ Ошибка при тесте: {e}")

print("\n🎉 Готово! Артефакты сохранены:")
print("   • model.pkl — модель для проверяющей системы")
print(f"   • {USER_PREFIX}_user_features — признаки пользователей в БД")
print(f"   • {USER_PREFIX}_post_features — признаки постов в БД")