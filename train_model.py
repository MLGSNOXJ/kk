import pandas as pd
import numpy as np
import torch
import psycopg2
from sqlalchemy import create_engine
from transformers import AutoTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
import os
import logging
import pickle

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
BATCH_SIZE = 16 # Уменьшил для стабильности на GPU/CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data_from_db():
    logger.info("📥 Загружаем данные...")
    conn = psycopg2.connect(DB_URL)
    
    user_data_sql = pd.read_sql("SELECT * FROM public.user_data", conn)
    post_data_sql = pd.read_sql("SELECT post_id, text, topic FROM public.post_text_df", conn)
    
    # ⚠️ ВАЖНО: ORDER BY RANDOM() убран, перемешиваем уже в pandas
    train_query = """
        SELECT user_id, post_id, target, timestamp
        FROM public.feed_data
        WHERE action = 'view' AND timestamp < '2021-12-22'
        LIMIT 1000000
    """
    test_query = """
        SELECT user_id, post_id, target, timestamp
        FROM public.feed_data
        WHERE action = 'view' AND timestamp >= '2021-12-22'
        LIMIT 200000
    """
    
    train_feed_data = pd.read_sql(train_query, conn).sample(frac=1, random_state=42).reset_index(drop=True)
    test_feed_data = pd.read_sql(test_query, conn).sample(frac=1, random_state=42).reset_index(drop=True)
    conn.close()
    
    return user_data_sql, post_data_sql, train_feed_data, test_feed_data

def create_user_features(user_data_sql, train_feed_data):
    logger.info("👤 Создаём фичи пользователей...")
    user_data = user_data_sql.copy()
    
    user_stats = train_feed_data.groupby('user_id').agg(
        total_views=('target', 'count'),
        total_likes=('target', 'sum')
    ).reset_index()
    
    # Защита от деления на 0
    user_stats['total_views_safe'] = user_stats['total_views'].replace(0, 1)
    user_stats['likes_freq'] = (user_stats['total_likes'] / user_stats['total_views_safe']).round(3)
    user_stats['daily_views'] = (user_stats['total_views'] / 10).round(3)
    
    user_data['russia'] = (user_data['country'] == 'Russia').astype(int)
    user_data['moscow'] = (user_data['city'] == 'Moscow').astype(int)
    user_data['saint_p'] = (user_data['city'] == 'Saint Petersburg').astype(int)
    user_data = user_data.drop(['city', 'country'], axis=1, errors='ignore')
    
    for col in ['exp_group', 'os', 'source']:
        if col in user_data.columns:
            user_data[col] = user_data[col].fillna('unknown').astype(str)
            ohe = pd.get_dummies(user_data[col], prefix=col, drop_first=True).astype(int)
            user_data = pd.concat([user_data.drop(col, axis=1), ohe], axis=1)
            
    user_data = user_data.merge(user_stats[['user_id', 'likes_freq', 'daily_views']], on='user_id', how='left')
    user_data = user_data.fillna(0)
    
    # Оптимизация памяти
    for col in user_data.select_dtypes(include=['int64']).columns:
        user_data[col] = user_data[col].astype('int32')
        
    logger.info(f"✅ Фичи пользователей: {user_data.shape}")
    return user_data

def get_bert_embeddings(text_series, batch_size=BATCH_SIZE):
    logger.info("🤖 Генерируем BERT-эмбеддинги...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased').to(DEVICE)
    model.eval()
    
    texts = text_series.fillna('').astype(str).tolist()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
            outputs = model(**encoded)
            
            mask = encoded['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_emb = torch.sum(outputs.last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            
            batch_emb = (sum_emb / sum_mask).cpu().numpy()
            embeddings.append(batch_emb)

    del model, tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return np.vstack(embeddings)

def create_post_features(post_data_sql, train_feed_data, bert_embeddings=None):
    logger.info("📝 Создаём фичи постов...")
    post_data = post_data_sql.copy()
    
    post_stats = train_feed_data.groupby('post_id').agg(
        total_views=('target', 'count'),
        total_likes=('target', 'sum')
    ).reset_index()
    
    post_stats['total_views_safe'] = post_stats['total_views'].replace(0, 1)
    post_stats['post_likes_freq'] = (post_stats['total_likes'] / post_stats['total_views_safe']).round(3)
    post_stats['post_daily_views'] = (post_stats['total_views'] / 10).round(3)
    
    post_data = post_data.merge(post_stats[['post_id', 'post_likes_freq', 'post_daily_views']], on='post_id', how='left')
    post_data = post_data.fillna(0)
    
    # TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(post_data['text'].fillna('').astype(str))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    post_data = pd.concat([post_data, tfidf_df], axis=1)
    
    post_data['text_length'] = post_data['text'].fillna('').str.len()
    
    # BERT + PCA + Clustering
    if bert_embeddings is not None:
        pca = PCA(n_components=15, random_state=42)
        emb_reduced = pca.fit_transform(bert_embeddings)
        
        cos_dist = cosine_distances(emb_reduced)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='precomputed', linkage='average')
        labels = clustering.fit_predict(cos_dist)
        
        for i in range(15):
            post_data[f'bert_pca_{i}'] = emb_reduced[:, i]
        post_data['bert_cluster'] = labels

    # OHE для категории
    for col in ['topic', 'bert_cluster']:
        if col in post_data.columns:
            post_data[col] = post_data[col].fillna('unknown').astype(str)
            ohe = pd.get_dummies(post_data[col], prefix=col, drop_first=True).astype(int)
            post_data = pd.concat([post_data.drop(col, axis=1), ohe], axis=1)

    post_data = post_data.drop(['text'], axis=1, errors='ignore')
    
    # Оптимизация памяти
    for col in post_data.select_dtypes(include=['int64']).columns:
        post_data[col] = post_data[col].astype('int32')
        
    logger.info(f"✅ Фичи постов: {post_data.shape}")
    return post_data

def main():
    try:
        user_data_sql, post_data_sql, train_feed_data, test_feed_data = load_data_from_db()
        
        user_data = create_user_features(user_data_sql, train_feed_data)
        bert_embeddings = get_bert_embeddings(post_data_sql['text'])
        post_data = create_post_features(post_data_sql, train_feed_data, bert_embeddings)
        
        # Merge выборки
        train_merged = train_feed_data.merge(user_data, on='user_id', how='left').merge(post_data, on='post_id', how='left')
        test_merged = test_feed_data.merge(user_data, on='user_id', how='left').merge(post_data, on='post_id', how='left')
        
        train_merged = train_merged.fillna(0)
        test_merged = test_merged.fillna(0)
        
        exclude_cols = ['target', 'user_id', 'post_id', 'timestamp']
        X_train = train_merged.drop(exclude_cols, axis=1)
        y_train = train_merged['target']
        X_test = test_merged.drop(exclude_cols, axis=1)
        y_test = test_merged['target']
        
        # Обучение
        pool_train = Pool(X_train, y_train)
        pool_test = Pool(X_test, y_test)
        
        model = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=8,
            loss_function='Logloss', eval_metric='AUC',
            early_stopping_rounds=50, verbose=100, 
            random_seed=42, thread_count=-1, allow_writing_files=False
        )
        model.fit(pool_train, eval_set=pool_test)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"\n✅ Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")

        # === СОХРАНЕНИЕ ===
        # 1. Модель
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # 2. Признаки в БД (ИМЕНА ТАБЛИЦ ДОЛЖНЫ СОВПАДАТЬ С APP.PY!)
        engine = create_engine(DB_URL)
        user_data.to_sql('user_features_final', engine, if_exists='replace', index=False, method='multi', chunksize=1000)
        post_data.to_sql('post_features_final', engine, if_exists='replace', index=False, method='multi', chunksize=1000)
        
        logger.info("💾 Всё сохранено!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main()