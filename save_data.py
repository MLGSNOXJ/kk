# save_data.py
import pandas as pd
from database import postgres_connection

conn = postgres_connection()
print("⏳ Загружаем данные из БД...")
df_users = pd.read_sql("SELECT * FROM public.user_data", conn)
df_posts = pd.read_sql("SELECT * FROM public.post_text_df", conn)
df_feed = pd.read_sql("SELECT * FROM public.feed_data LIMIT 5000000", conn)
conn.close()

df_users.to_parquet("data/users.parquet", index=False)
df_posts.to_parquet("data/posts.parquet", index=False)
df_feed.to_parquet("data/feed.parquet", index=False)
print("✅ Данные успешно сохранены в папку data/")