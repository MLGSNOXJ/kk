import pandas as pd
import time
from database import postgres_connection

def load_data_from_db(max_retries: int = 3):
    """Подключается к PostgreSQL и выгружает 3 таблицы в pandas DataFrames."""
    for attempt in range(max_retries):
        try:
            conn = postgres_connection()
            print("✅ Подключение к PostgreSQL установлено")

            # 1. Таблица пользователей
            print("⏳ Загрузка пользователей...")
            df_users = pd.read_sql("SELECT * FROM public.user_data", conn)
            print(f"✅ user_data: {df_users.shape}")

            # 2. Таблица постов
            print("⏳ Загрузка постов...")
            df_posts = pd.read_sql("SELECT * FROM public.post_text_df", conn)
            print(f"✅ post_text_df: {df_posts.shape}")

            # 3. История взаимодействий (лимит 5 млн строк)
            print("⏳ Загрузка feed_data (LIMIT 5 000 000)...")
            feed_query = """
                SELECT timestamp, user_id, post_id, action, target
                FROM public.feed_data
                LIMIT 5000000
            """
            df_feed = pd.read_sql(feed_query, conn)
            print(f"✅ feed_data: {df_feed.shape}")

            conn.close()
            print("🔌 Подключение закрыто")
            return df_users, df_posts, df_feed

        except Exception as e:
            print(f"⚠️ Ошибка при загрузке (попытка {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("⏳ Ждём 5 секунд и пробуем снова...")
                time.sleep(5)
            else:
                raise e

# Запуск выгрузки
df_users, df_posts, df_feed = load_data_from_db()