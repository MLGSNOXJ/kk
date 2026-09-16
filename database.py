import psycopg2


def postgres_connection():
    """Устанавливает и возвращает соединение с PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host="postgres.lab.karpov.courses",
            port=6432,
            database="startml",
            user="robot-startml-ro",
            password="pheiph0hahj1Vaif",
        )
    except Exception as e:
        print("❌ Ошибка при подключении к базе данных.")
        raise e

    conn.autocommit = True

    return conn
