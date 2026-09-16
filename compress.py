# compress.py — сжимает модель до <100 МБ
import joblib
import os

# Пути
SRC = "data/model.pkl"      # ваша большая модель (или где она у вас лежит)
DST = "model.pkl"           # финальный файл для отправки

print(f"📦 Загружаем модель из {SRC}...")
model = joblib.load(SRC)

print("🗜️ Сжимаем и сохраняем...")
joblib.dump(model, DST, compress=3)  # compress=3 — оптимальное сжатие

# Проверка
size_mb = os.path.getsize(DST) / (1024 * 1024)
print(f"✅ Готово! Новый размер: {size_mb:.2f} МБ")

if size_mb < 100:
    print("🎉 Файл готов к отправке!")
else:
    print(f"⚠️ Всё ещё {size_mb:.2f} МБ > 100 МБ. См. решение ниже.")