import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_user_features(df_users: pd.DataFrame, df_feed: pd.DataFrame) -> pd.DataFrame:
    """Создаёт признаки пользователей на основе профиля и истории активности."""
    
    # 1. Базовые признаки из профиля
    user_features = df_users.copy()
    
    # 2. Агрегированные признаки из истории действий
    # Фильтруем только просмотры для статистики (лайки учтём отдельно)
    feed_views = df_feed[df_feed['action'] == 'view'].copy()
    
    # Статистика по пользователю: сколько всего просмотрел, лайкнул, CTR
    user_stats = feed_views.groupby('user_id').agg(
        user_total_views=('post_id', 'count'),
        user_total_likes=('target', 'sum'),  # target=1 только для лайков после просмотра
    ).reset_index()
    
    user_stats['user_ctr'] = user_stats['user_total_likes'] / (user_stats['user_total_views'] + 1)  # +1 для избегания деления на 0
    
    # 3. Признаки активности по времени
    feed_views['hour'] = pd.to_datetime(feed_views['timestamp']).dt.hour
    feed_views['dayofweek'] = pd.to_datetime(feed_views['timestamp']).dt.dayofweek
    
    user_time_stats = feed_views.groupby('user_id').agg(
        user_avg_hour=('hour', 'mean'),
        user_avg_dayofweek=('dayofweek', 'mean'),
        user_active_hours=('hour', lambda x: x.nunique()),  # разнообразие часов активности
    ).reset_index()
    
    # 4. Предпочтения по темам (какие темы пользователь чаще лайкает)
    # Объединяем с постами, чтобы получить topic
    feed_with_topic = feed_views.merge(df_posts[['id', 'topic']], left_on='post_id', right_on='id', how='left')
    
    user_topic_pref = feed_with_topic.groupby(['user_id', 'topic']).agg(
        topic_likes=('target', 'sum')
    ).reset_index()
    
    # Берём топ-1 любимую тему пользователя
    user_top_topic = user_topic_pref.loc[user_topic_pref.groupby('user_id')['topic_likes'].idxmax()][['user_id', 'topic']]
    user_top_topic.rename(columns={'topic': 'user_favorite_topic'}, inplace=True)
    
    # 5. Собираем все пользовательские признаки
    user_features = user_features.merge(user_stats, on='user_id', how='left')
    user_features = user_features.merge(user_time_stats, on='user_id', how='left')
    user_features = user_features.merge(user_top_topic, on='user_id', how='left')
    
    # Заполняем пропуски (для новых пользователей)
    user_features['user_total_views'] = user_features['user_total_views'].fillna(0)
    user_features['user_total_likes'] = user_features['user_total_likes'].fillna(0)
    user_features['user_ctr'] = user_features['user_ctr'].fillna(0)
    
    return user_features


def create_post_features(df_posts: pd.DataFrame, df_feed: pd.DataFrame) -> pd.DataFrame:
    """Создаёт признаки постов на основе контента и популярности."""
    
    post_features = df_posts.copy()
    
    # 1. Признаки текста (простые статистики)
    post_features['text_length'] = post_features['text'].astype(str).str.len()
    post_features['word_count'] = post_features['text'].astype(str).str.split().str.len()
    post_features['unique_words'] = post_features['text'].astype(str).apply(lambda x: len(set(x.lower().split())))
    
    # 2. Популярность поста (на основе истории лайков)
    feed_likes = df_feed[(df_feed['action'] == 'view') & (df_feed['target'] == 1)].copy()
    
    post_popularity = feed_likes.groupby('post_id').agg(
        post_total_likes=('target', 'count'),
        post_unique_viewers=('user_id', 'nunique'),
    ).reset_index()
    
    post_popularity['post_like_rate'] = post_popularity['post_total_likes'] / (post_popularity['post_unique_viewers'] + 1)
    
    post_features = post_features.merge(post_popularity, left_on='id', right_on='post_id', how='left')
    
    # Заполняем пропуски для новых постов
    post_features['post_total_likes'] = post_features['post_total_likes'].fillna(0)
    post_features['post_unique_viewers'] = post_features['post_unique_viewers'].fillna(0)
    post_features['post_like_rate'] = post_features['post_like_rate'].fillna(0)
    
    return post_features


def create_temporal_features(df_feed: pd.DataFrame) -> pd.DataFrame:
    """Извлекает признаки из временной метки взаимодействия."""
    
    df = df_feed.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Базовые временные признаки
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    # Время суток (категориальный признак)
    def get_time_slot(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df['time_slot'] = df['hour'].apply(get_time_slot)
    
    return df


def prepare_training_data(df_users: pd.DataFrame, df_posts: pd.DataFrame, df_feed: pd.DataFrame):
    """Основная функция: создаёт признаки, объединяет данные и готовит выборку."""
    
    print("🔧 Создаём признаки пользователей...")
    user_features = create_user_features(df_users, df_feed)
    
    print("🔧 Создаём признаки постов...")
    post_features = create_post_features(df_posts, df_feed)
    
    print("🔧 Извлекаем временные признаки...")
    feed_with_time = create_temporal_features(df_feed)
    
    # 🔥 Критически важно: для обучения используем ТОЛЬКО просмотры (action == 'view')
    # У лайков target = NaN, они не подходят для обучения классификации
    df_train = feed_with_time[feed_with_time['action'] == 'view'].copy()
    print(f"✅ Фильтруем просмотры: {len(df_train)} строк для обучения")
    
    # Объединяем с признаками пользователей и постов
    print("🔗 Объединяем данные...")
    df_train = df_train.merge(user_features, on='user_id', how='left')
    df_train = df_train.merge(post_features, left_on='post_id', right_on='id', how='left', suffixes=('', '_post'))
    
    # Удаляем дублирующиеся колонки
    if 'id_post' in df_train.columns:
        df_train.drop(columns=['id_post'], inplace=True)
    
    # 🔤 Кодирование категориальных признаков
    categorical_cols = ['gender', 'country', 'city', 'os', 'source', 'topic', 'user_favorite_topic', 'time_slot']
    
    # Используем Target Encoding для категориальных признаков (хорошо работает с деревьями)
    # Важно: кодируем ТОЛЬКО на тренировочной части, чтобы не было утечки
    encoder = TargetEncoder(cols=categorical_cols, smoothing=10)
    
    # Для простоты сначала заполним пропуски в категориях
    for col in categorical_cols:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna('unknown')
    
    # Разделяем признаки и таргет
    feature_cols = [col for col in df_train.columns if col not in ['user_id', 'post_id', 'timestamp', 'action', 'target']]
    
    # Отбираем только числовые и закодированные признаки
    X = df_train[feature_cols].copy()
    y = df_train['target'].copy()
    
    # Заполняем оставшиеся пропуски в числовых признаках
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    # 🎯 Разделение на train/test
    # Вариант 1: Временной сплит (рекомендуется для рекомендаций)
    df_train_sorted = df_train.sort_values('timestamp')
    split_idx = int(len(df_train_sorted) * 0.8)
    
    train_idx = df_train_sorted.index[:split_idx]
    test_idx = df_train_sorted.index[split_idx:]
    
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    print(f"\n✅ Обучающая выборка: {X_train.shape}, Тестовая: {X_test.shape}")
    print(f"✅ Распределение target в train: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"✅ Распределение target в test: {y_test.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_test, y_train, y_test, feature_cols, encoder


# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    print("📥 Загружаем данные...")
    df_users = pd.read_parquet("data/users.parquet")
    df_posts = pd.read_parquet("data/posts.parquet") 
    df_feed = pd.read_parquet("data/feed.parquet")
    
    # 🔧 Приводим названия столбцов к формату, который ждёт наш код
    if 'post_id' in df_posts.columns and 'id' not in df_posts.columns:
        df_posts.rename(columns={'post_id': 'id'}, inplace=True)
        
    if 'user_id' not in df_users.columns and 'id' in df_users.columns:
        df_users.rename(columns={'id': 'user_id'}, inplace=True)
        
    print(f"✅ Колонки df_users: {df_users.columns.tolist()[:3]}...")
    print(f"✅ Колонки df_posts: {df_posts.columns.tolist()}")
    print(f"✅ Колонки df_feed: {df_feed.columns.tolist()[:5]}...")
    
    # Создаём выборку
    X_train, X_test, y_train, y_test, feature_cols, encoder = prepare_training_data(
        df_users, df_posts, df_feed
    )
        # 💾 Сохраняем результаты для следующего шага
    print("\n💾 Сохраняем данные...")
    X_train.to_parquet("data/X_train.parquet", index=False)
    X_test.to_parquet("data/X_test.parquet", index=False)
    
    # Series нужно превратить в DataFrame перед сохранением в parquet
    y_train.to_frame("target").to_parquet("data/y_train.parquet", index=False)
    y_test.to_frame("target").to_parquet("data/y_test.parquet", index=False)
    
    # Сохраняем список признаков
    pd.DataFrame({'feature': feature_cols}).to_csv("data/feature_cols.csv", index=False)
    
    print("✅ Готово! Данные сохранены в папке data/")