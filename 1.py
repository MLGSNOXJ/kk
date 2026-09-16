import pandas as pd
from sqlalchemy import create_engine

# Подключение к БД
DB_URL = "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml"
engine = create_engine(DB_URL)

# Период эксперимента (как в test_query из train_model.py)
EXPERIMENT_START_DATE = '2021-12-22'

print("Шаг 2: Загружаем датафрейм с просмотрами для получения пользователей и их групп...")
views_query = f"""
    SELECT f.user_id, u.exp_group
    FROM public.feed_data f
    JOIN public.user_data u ON f.user_id = u.user_id
    WHERE f.action = 'view' AND f.timestamp >= '{EXPERIMENT_START_DATE}'
"""
df_views = pd.read_sql(views_query, engine)

# Делаем табличку, у кого какая группа, используя .first()
user_groups = df_views.groupby('user_id')['exp_group'].first().reset_index()
print(f"   Найдено уникальных пользователей с просмотрами: {len(user_groups)}")

print("Шаг 1: Считаем количество лайков на пользователя...")
likes_query = f"""
    SELECT user_id, COUNT(*) as likes_count
    FROM public.feed_data
    WHERE action = 'like' AND timestamp >= '{EXPERIMENT_START_DATE}'
    GROUP BY user_id
"""
df_likes = pd.read_sql(likes_query, engine)
print(f"   Найдено пользователей, сделавших хотя бы один лайк: {len(df_likes)}")

print("Шаг 3: Объединяем датафреймы (Left Join)...")
# how='left' гарантирует, что останутся ВСЕ пользователи из user_groups (с просмотрами),
# даже если их нет в df_likes (у них будет NaN)
merged_df = user_groups.merge(df_likes, on='user_id', how='left')

print("Шаг 4: Заполняем пропуски нулями...")
# Это как раз те люди, у которых были просмотры, но не было лайков
merged_df['likes_count'] = merged_df['likes_count'].fillna(0)

print("Шаг 5: Считаем пропорцию...")
total_users = len(merged_df)
users_with_likes = (merged_df['likes_count'] > 0).sum()
proportion = users_with_likes / total_users

print("-" * 50)
print(f"Всего пользователей с просмотрами: {total_users}")
print(f"Пользователей с хотя бы одним лайком: {users_with_likes}")
print(f"✅ ИТОГОВАЯ ДОЛЯ: {proportion:.1%} (или {proportion:.4f})")
print("-" * 50)