# prov.py
from load_data import load_data_from_db  # импортируем функцию загрузки

# Вызываем функцию и получаем данные
df_users, df_posts, df_feed = load_data_from_db()

# Теперь можно использовать
print(df_users.head())
print(df_posts.info())
print(df_feed['action'].value_counts())