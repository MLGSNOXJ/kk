# compress_model.py
import joblib
import os

INPUT_MODEL = "data/model.pkl"  # или путь к вашей большой модели
OUTPUT_MODEL = "model.pkl"       # финальный файл для отправки

print(f"📦 Сжимаем {INPUT_MODEL}...")

# Загружаем модель
model = joblib.load(INPUT_MODEL)

# Сохраняем со сжатием (уровень 3 — хороший баланс скорости/размера)
joblib.dump(model, OUTPUT_MODEL, compress=3)

# Проверяем размер
size_mb = os.path.getsize(OUTPUT_MODEL) / (1024 * 1024)
print(f"✅ Готово! Размер: {size_mb:.2f} МБ")

if size_mb > 100:
    print("⚠️ Всё ещё больше 100 МБ. Попробуйте решение 2 ниже.")
else:
    print("🎉 Модель готова к отправке!")