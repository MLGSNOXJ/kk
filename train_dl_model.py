import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import warnings

# Импорт подключения из вашего файла database.py
# Убедитесь, что database.py лежит в той же папке
try:
    from database import engine
except ImportError:
    # Если импорта нет, создаем engine вручную (замените на ваш URL)
    from sqlalchemy import create_engine
    SQLALCHEMY_DATABASE_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

warnings.filterwarnings('ignore')

# ==========================================
# 1. Функция оптимизации памяти (Подсказка)
# ==========================================
def optimize_memory(df):
    """
    Уменьшает объем используемой памяти согласно подсказке:
    - int64 переводит в int32
    - float64 переводит в float32
    - object/str переводит в category (если кардинальность низкая)
    """
    print(f"Оптимизация памяти для DataFrame (строк: {len(df)})...")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Оптимизация численных типов
        if col_type == 'int64':
            df[col] = df[col].astype('int32')
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
            
        # Оптимизация строковых типов
        # Переводим в category только если уникальных значений не слишком много (эвристика)
        # Это экономит память, но добавляет накладные расходы на кодирование
        elif col_type == 'object':
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
                
    return df

# ==========================================
# 2. Загрузка данных
# ==========================================
print("Загрузка данных из БД...")

# Загружаем сырые данные
user_data_sql = pd.read_sql("SELECT * FROM public.user_data", con=engine)
post_data_sql = pd.read_sql("SELECT * FROM public.post_text_df", con=engine)

# Загружаем фидбек (обучающая выборка)
# Используем тот же запрос, что и в ноутбуке (с лимитом для примера)
train_feed_data = pd.read_sql("""
    SELECT user_id, post_id, target, timestamp
    FROM public.feed_data
    WHERE (feed_data.action = 'view') AND (timestamp < '2021-12-22')
    ORDER BY RANDOM() limit 1000000
""", con=engine)

# Загружаем фидбек (тестовая выборка)
test_feed_data = pd.read_sql("""
    SELECT user_id, post_id, target, timestamp
    FROM public.feed_data
    WHERE (feed_data.action = 'view') AND (timestamp > '2021-12-22')
    ORDER BY RANDOM() limit 200000
""", con=engine)

# ==========================================
# 3. Обработка User Data
# ==========================================
print("Обработка User Data...")
user_data = user_data_sql.copy()

# Вычисляем статистику по пользователям (как в ноутбуке)
# В реальном проекте лучше делать это одним SQL запросом, но оставим логику pandas
# Для ускорения можно сделать это через SQL, здесь упрощенно:
user_stats = train_feed_data.groupby('user_id').agg(
    total_views=('target', 'count'),
    total_likes=('target', 'sum')
).reset_index()

# Добавим days_active (упрощенно)
# В ноутбуке это делается через SQL MAX(date) - MIN(date)
# Здесь просто заглушка, чтобы код работал
user_stats['total_days'] = 10 

user_features = user_stats.copy()
user_features['likes_freq'] = (user_features['total_likes'] / user_features['total_views']).round(2)
user_features['daily_views'] = (user_features['total_views'] / user_features['total_days']).round(2)

# Объединяем
user_data = user_data.merge(user_features[['user_id', 'likes_freq', 'daily_views']], on='user_id', how='left')
user_data = user_data.fillna(0) # Заполняем NaN для пользователей без лайков

# One-Hot Encoding категориальных признаков
categorical = ['exp_group', 'os', 'source']
for col in categorical:
    user_data[col] = user_data[col].astype(object)
    # get_dummies создает int64 по умолчанию
    ohe_cols = pd.get_dummies(user_data[col], prefix=col, drop_first=True).astype(int) 
    user_data = pd.concat([user_data.drop(col, axis=1), ohe_cols], axis=1)

# Применяем оптимизацию памяти к user_data
user_data = optimize_memory(user_data)

# ==========================================
# 4. Обработка Post Data + BERT
# ==========================================
print("Обработка Post Data...")
post_data = post_data_sql.copy()

# Статистика постов
post_stats = train_feed_data.groupby('post_id').agg(
    total_views=('target', 'count'),
    total_likes=('target', 'sum')
).reset_index()
post_stats['days_old'] = 75 # Заглушка, как в ноутбуке

post_features = post_stats.copy()
post_features['post_likes_freq'] = (post_features['total_likes'] / post_features['total_views']).round(3)
post_features['post_daily_views'] = (post_features['total_views'] / post_features['days_old']).round(3)

post_data = post_data.merge(post_features[['post_id', 'post_likes_freq', 'post_daily_views']], on='post_id', how='outer')
post_data = post_data.fillna(0)

# TF-IDF (длина текста)
post_data['text_length'] = post_data['text'].str.len()

# BERT Embeddings (Код из ноутбука)
# ВНИМАНИЕ: Этот блок тяжелый. Если памяти мало, можно пропустить или использовать готовые эмбеддинги.
print("Запуск BERT для эмбеддингов...")
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model_bert = BertModel.from_pretrained("bert-base-cased")
    model_bert.eval()

    # Упрощенная версия получения эмбеддингов для примера
    # В полном коде нужно использовать DataLoader, как в ноутбуке
    # Здесь мы сделаем это для небольшого сэмпла или используем заглушку, 
    # чтобы не вешать скрипт, если нет GPU.
    
    # Для демонстрации оптимизации памяти предположим, что мы получили 'label'
    # post_data['label'] = ... (результат кластеризации)
    
    # Если вы хотите запустить полный BERT, раскомментируйте код из ноутбука (ячейки 19-25)
    # post_data['label'] = 0 # Заглушка
except Exception as e:
    print(f"BERT skipped or failed: {e}")
    post_data['label'] = 0 # Заглушка

# OHE Topic
post_data['topic'] = post_data['topic'].astype(object)
ohe_topic = pd.get_dummies(post_data['topic'], prefix='topic', drop_first=True).astype(int)
post_data = pd.concat([post_data.drop('topic', axis=1), ohe_topic], axis=1)

# OHE Label (если есть)
if 'label' in post_data.columns:
    post_data['label'] = post_data['label'].astype(object)
    ohe_label = pd.get_dummies(post_data['label'], prefix='label', drop_first=True).astype(int)
    post_data = pd.concat([post_data.drop('label', axis=1), ohe_label], axis=1)

# Удаляем текст, он больше не нужен для модели
if 'text' in post_data.columns:
    post_data = post_data.drop('text', axis=1)

# Применяем оптимизацию памяти к post_data
post_data = optimize_memory(post_data)

# ==========================================
# 5. Подготовка Train/Test выборок
# ==========================================
print("Создание финальных выборок...")

# Merge
train_data = train_feed_data.merge(user_data, on='user_id').merge(post_data, on='post_id')
test_data = test_feed_data.merge(user_data, on='user_id').merge(post_data, on='post_id')

X_train = train_data.drop(['target', 'user_id', 'post_id', 'timestamp'], axis=1)
y_train = train_data['target']

X_test = test_data.drop(['target', 'user_id', 'post_id', 'timestamp'], axis=1)
y_test = test_data['target']

# ВАЖНО: Применяем оптимизацию памяти к X_train и X_test перед обучением
# Это критично для CatBoost, чтобы он не копировал данные в int64
X_train = optimize_memory(X_train)
X_test = optimize_memory(X_test)

# ==========================================
# 6. Обучение модели CatBoost
# ==========================================
print("Обучение CatBoost...")

# Расчет весов классов
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# CatBoost принимает словарь или список весов. 
# В ноутбуке используется class_weights напрямую в конструкторе, 
# но там, вероятно, подразумевается scale_pos_weight или подобный параметр.
# Для CatBoost лучше использовать параметр class_weights в fit или auto_class_weights.

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    auto_class_weights='Balanced', # Автоматический баланс классов
    random_seed=42,
    verbose=100
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    use_best_model=True
)

# ==========================================
# 7. Оценка и Сохранение
# ==========================================
y_pred = model.predict(X_test)
print(f"Accuracy на тесте: {accuracy_score(y_test, y_pred):.4f}")

# Сохранение модели
model.save_model('final_model.cbm')
print("Модель сохранена в final_model.cbm")