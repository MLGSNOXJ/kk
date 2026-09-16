import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from table_feed import Feed, Post, User
from schema import UserGet, PostGet, FeedGet, Response
import hashlib
from catboost import CatBoostClassifier
import logging

# ==================== КОНСТАНТЫ ====================
SALT = 'salt_2025'
TOP_K = 5  # Количество рекомендаций для оценки

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_exp_group(user_id: int) -> str:
    """Определение экспериментальной группы пользователя по хешу"""
    value_str = str(user_id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    return 'control' if value_num % 100 <= 50 else 'test'


def load_model(exp_group: str, model_paths: Dict[str, str]) -> CatBoostClassifier:
    """Загрузка модели для соответствующей экспериментальной группы"""
    model_path = model_paths.get(exp_group)
    if not model_path:
        raise ValueError(f"Model path not found for group: {exp_group}")
    
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загрузка фичей пользователей и постов"""
    user_data = pd.read_sql(
        'SELECT * FROM n_habenko_14_users_lesson_22', 
        con=engine
    )
    post_data = pd.read_sql(
        'SELECT * FROM n_habenko_14_posts_lesson_22', 
        con=engine
    )
    posts = pd.read_sql(
        'SELECT * FROM public.post_text_df', 
        con=engine
    )
    return user_data, post_data, posts


# ==================== ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ ====================

def get_recommendations(
    user_id: int, 
    model: CatBoostClassifier, 
    user_data: pd.DataFrame, 
    post_data: pd.DataFrame, 
    posts: pd.DataFrame, 
    top_k: int = TOP_K
) -> List[int]:
    """
    Получение топ-K рекомендаций для пользователя
    
    Returns:
        List[int]: Список ID постов в порядке убывания релевантности
    """
    # Создаём матрицу признаков: пользователь × все посты
    user_features = user_data[user_data['user_id'] == user_id].copy()
    if user_features.empty:
        return []
    
    # Cross join для создания всех возможных пар пользователь-пост
    data = user_features.merge(post_data, how='cross')
    
    # Удаляем служебные колонки
    cols_to_drop = ['user_id', 'post_id']
    data = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)
    
    if data.empty:
        return []
    
    # Получаем предсказания вероятности клика
    predictions = model.predict_proba(data)[:, 1]
    
    # Сортируем по убыванию вероятности и берём топ-K
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    recommended_post_ids = posts.iloc[top_indices]['post_id'].tolist()
    
    return recommended_post_ids


# ==================== МЕТРИКИ ====================

def hit_at_k(actual_post_id: int, recommended_post_ids: List[int], k: int = TOP_K) -> float:
    """
    Binary hit@K: 1 если релевантный пост в топ-K, иначе 0
    """
    return 1.0 if actual_post_id in recommended_post_ids[:k] else 0.0


def calculate_hitrate_batch(
    test_interactions: pd.DataFrame,
    user_data: pd.DataFrame,
    post_data: pd.DataFrame,
    posts: pd.DataFrame,
    model_control: CatBoostClassifier,
    model_test: CatBoostClassifier,
    top_k: int = TOP_K
) -> Dict[str, float]:
    """
    Расчёт HitRate@K для батча тестовых взаимодействий
    
    Parameters:
    -----------
    test_interactions : pd.DataFrame
        DataFrame с колонками: user_id, post_id (фактические взаимодействия)
    
    Returns:
    --------
    dict : Словарь с метриками hitrate
    """
    hits = []
    hits_by_group = {'control': [], 'test': []}
    
    for idx, row in test_interactions.iterrows():
        user_id = row['user_id']
        actual_post_id = row['post_id']
        
        # Определяем группу и загружаем модель
        exp_group = get_exp_group(user_id)
        model = model_control if exp_group == 'control' else model_test
        
        # Получаем рекомендации
        recommended_ids = get_recommendations(
            user_id, model, user_data, post_data, posts, top_k
        )
        
        if not recommended_ids:
            continue
            
        # Проверяем hit
        hit = hit_at_k(actual_post_id, recommended_ids, top_k)
        hits.append(hit)
        hits_by_group[exp_group].append(hit)
    
    # Агрегируем результаты
    results = {
        'overall_hitrate': np.mean(hits) if hits else 0.0,
        'total_samples': len(hits),
        'total_hits': int(sum(hits)),
        'hitrate_by_group': {}
    }
    
    for group in hits_by_group:
        group_hits = hits_by_group[group]
        if group_hits:
            results['hitrate_by_group'][group] = np.mean(group_hits)
        else:
            results['hitrate_by_group'][group] = 0.0
    
    return results


def calculate_ndcg_at_k(
    actual_post_ids: List[int], 
    recommended_post_ids: List[int], 
    k: int = TOP_K
) -> float:
    """
    Расчёт NDCG@K (Normalized Discounted Cumulative Gain)
    
    Parameters:
    -----------
    actual_post_ids : List[int] - релевантные посты для пользователя
    recommended_post_ids : List[int] - рекомендованные посты (упорядочены)
    k : int - cutoff для оценки
    
    Returns:
    --------
    float : NDCG@K score в диапазоне [0, 1]
    """
    def dcg(relevant: set, recommended: List[int], k: int) -> float:
        score = 0.0
        for i, post_id in enumerate(recommended[:k]):
            if post_id in relevant:
                score += 1.0 / np.log2(i + 2)  # i+2 т.к. индексация с 0
        return score
    
    def idcg(n_relevant: int, k: int) -> float:
        return sum(1.0 / np.log2(i + 2) for i in range(min(n_relevant, k)))
    
    relevant_set = set(actual_post_ids)
    n_relevant = len(relevant_set)
    
    dcg_score = dcg(relevant_set, recommended_post_ids, k)
    idcg_score = idcg(n_relevant, k)
    
    return dcg_score / idcg_score if idcg_score > 0 else 0.0


def calculate_mrr(
    actual_post_id: int, 
    recommended_post_ids: List[int], 
    k: int = TOP_K
) -> float:
    """
    Mean Reciprocal Rank: 1/rank первого релевантного элемента
    """
    for rank, post_id in enumerate(recommended_post_ids[:k], start=1):
        if post_id == actual_post_id:
            return 1.0 / rank
    return 0.0


# ==================== ЗАГРУЗКА ДАННЫХ ИЗ БД ====================

def get_test_interactions(
    user_ids: List[int] = None,
    limit: int = 1000,
    min_interactions_per_user: int = 1
) -> pd.DataFrame:
    """
    Загрузка тестовых взаимодействий из таблицы Feed
    
    Parameters:
    -----------
    user_ids : List[int], optional - список пользователей для оценки
    limit : int - максимальное количество записей
    min_interactions_per_user : int - мин. кол-во взаимодействий на пользователя
    
    Returns:
    --------
    pd.DataFrame с колонками user_id, post_id
    """
    with SessionLocal() as db:
        query = db.query(Feed.user_id, Feed.post_id).order_by(Feed.time.desc())
        
        if user_ids:
            query = query.filter(Feed.user_id.in_(user_ids))
        
        interactions = pd.read_sql(query.statement, query.session.bind)
    
    # Фильтрация пользователей с достаточным количеством взаимодействий
    if min_interactions_per_user > 1:
        user_counts = interactions['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions_per_user].index
        interactions = interactions[interactions['user_id'].isin(valid_users)]
    
    # Сэмплирование если нужно
    if len(interactions) > limit:
        interactions = interactions.sample(n=limit, random_state=42)
    
    return interactions.reset_index(drop=True)


# ==================== ОТЧЁТНОСТЬ ====================

def print_evaluation_report(results: Dict[str, float], top_k: int = TOP_K):
    """Вывод форматированного отчёта по метрикам"""
    print("\n" + "=" * 70)
    print(f"ОТЧЁТ ПО ОЦЕНКЕ РЕКОМЕНДАЦИЙ (HitRate@{top_k})")
    print("=" * 70)
    print(f"Всего оценено взаимодействий: {results.get('total_samples', 0)}")
    print(f"Всего попаданий (hits): {results.get('total_hits', 0)}")
    print(f"\n📊 OVERALL HitRate@{top_k}: {results['overall_hitrate']:.4f} "
          f"({results['overall_hitrate']*100:.2f}%)")
    
    if 'hitrate_by_group' in results:
        print(f"\n📈 HitRate по экспериментальным группам:")
        for group, hitrate in results['hitrate_by_group'].items():
            print(f"   • {group}: {hitrate:.4f} ({hitrate*100:.2f}%)")
    
    print("=" * 70 + "\n")


def save_results_to_file(results: Dict, filepath: str = 'hitrate_results.csv'):
    """Сохранение результатов в CSV файл"""
    df = pd.DataFrame([{
        'metric': 'overall_hitrate',
        'value': results['overall_hitrate']
    }, {
        'metric': 'total_samples',
        'value': results['total_samples']
    }, {
        'metric': 'total_hits',
        'value': results['total_hits']
    }])
    
    for group, hitrate in results.get('hitrate_by_group', {}).items():
        df = pd.concat([df, pd.DataFrame([{
            'metric': f'hitrate_{group}',
            'value': hitrate
        }])], ignore_index=True)
    
    df.to_csv(filepath, index=False)
    print(f"✅ Результаты сохранены в {filepath}")


# ==================== MAIN ФУНКЦИЯ ====================

def evaluate_recommendations(
    model_paths: Dict[str, str],
    limit: int = 500,
    top_k: int = TOP_K,
    user_ids: List[int] = None,
    save_results: bool = True
) -> Dict[str, float]:
    """
    Основная функция для оценки качества рекомендаций
    
    Parameters:
    -----------
    model_paths : dict
        Пути к моделям: {'control': 'path/to/model', 'test': 'path/to/model'}
    limit : int
        Количество тестовых взаимодействий для оценки
    top_k : int
        Количество рекомендаций для расчёта метрик
    user_ids : List[int], optional
        Список пользователей для оценки (None = все)
    save_results : bool
        Сохранять ли результаты в файл
    
    Returns:
    --------
    dict : Словарь с метриками качества
    """
    print(f"🚀 Запуск оценки рекомендаций (HitRate@{top_k})...")
    
    # 1. Загрузка данных
    print("📦 Загрузка фичей...")
    user_data, post_data, posts = load_features()
    
    # 2. Загрузка моделей
    print("🤖 Загрузка моделей...")
    model_control = load_model('control', model_paths)
    model_test = load_model('test', model_paths)
    
    # 3. Загрузка тестовых взаимодействий
    print(f"📋 Загрузка {limit} тестовых взаимодействий...")
    test_interactions = get_test_interactions(
        user_ids=user_ids, 
        limit=limit
    )
    print(f"   Загружено {len(test_interactions)} взаимодействий")
    
    # 4. Расчёт метрик
    print("🔍 Расчёт HitRate...")
    results = calculate_hitrate_batch(
        test_interactions=test_interactions,
        user_data=user_data,
        post_data=post_data,
        posts=posts,
        model_control=model_control,
        model_test=model_test,
        top_k=top_k
    )
    
    # 5. Вывод отчёта
    print_evaluation_report(results, top_k)
    
    # 6. Сохранение результатов
    if save_results:
        save_results_to_file(results)
    
    return results


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    # Пути к сохранённым моделям
    MODEL_PATHS = {
        'control': '/workdir/user_input/model_control',  # или локальный путь
        'test': '/workdir/user_input/model_test'
    }
    
    # Запуск оценки
    results = evaluate_recommendations(
        model_paths=MODEL_PATHS,
        limit=500,      # количество тестовых примеров
        top_k=5,        # оцениваем топ-5 рекомендаций
        save_results=True
    )
    
    # Пример расчёта дополнительных метрик для одного пользователя
    print("\n🔬 Пример расчёта дополнительных метрик:")
    user_id = 200
    actual_post_id = 12345  # пост, с которым пользователь реально взаимодействовал
    
    exp_group = get_exp_group(user_id)
    model = load_model(exp_group, MODEL_PATHS)
    user_data, post_data, posts = load_features()
    
    recs = get_recommendations(user_id, model, user_data, post_data, posts, top_k=5)
    
    print(f"User {user_id} (group: {exp_group})")
    print(f"Actual post: {actual_post_id}")
    print(f"Recommended: {recs}")
    print(f"Hit@5: {hit_at_k(actual_post_id, recs, k=5)}")
    print(f"MRR@5: {calculate_mrr(actual_post_id, recs, k=5)}")
    print(f"NDCG@5: {calculate_ndcg_at_k([actual_post_id], recs, k=5):.4f}")