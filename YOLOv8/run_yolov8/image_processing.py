import os
import shutil
from pathlib import Path
import cv2
from ultralytics import YOLO
from datetime import datetime
import openpyxl
from tqdm import tqdm  # для прогрес-барів

# =============================
# Налаштування
# =============================
IMAGES_DIR = Path("images")  # Вхідна тека з фото
OUTPUT_DIR = Path("processed_images")  # Вихідна тека для оброблених фото
MODEL_PATH = Path("best.pt")  # Вага моделі YOLOv8
IMG_SIZE = 1024  # Розмір зображення для YOLO

# Класи (припускаємо, що 0=container, 1=stem)
CLASS_CONTAINER = 0
CLASS_STEM = 1

# =============================
# Завантаження моделі
# =============================
print("Завантаження моделі...")
try:
    model = YOLO(str(MODEL_PATH))  # Завантаження моделі (автоматично використовує GPU якщо доступний)
    print("✅ Модель успішно завантажено")
except Exception as e:
    print(f"❌ Помилка завантаження моделі: {e}")
    exit(1)

# =============================
# Функція: отримати дату з імені файлу
# =============================
def extract_date_from_filename(filename: str) -> str:
    """Витягує дату з імені файлу типу IMG_20250716_023515.jpg Повертає строку у форматі dd.mm.yy"""
    parts = filename.split("_")
    for p in parts:
        if len(p) == 8 and p.isdigit():
            dt = datetime.strptime(p, "%Y%m%d")
            return dt.strftime("%d.%m.%y")
    return None


# =============================
# Функція: малювання результатів
# =============================
def process_image(image_path: Path, output_path: Path):
    """Обробляє одне фото: детектує об'єкти, малює рамки, рахує стеблини. Зберігає результат у output_path."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Не вдалося завантажити зображення: {image_path}")
        return 0
        
    results = model.predict(source=str(image_path), imgsz=IMG_SIZE, verbose=False)[0]
    stems = []
    containers = []

    # Збір координат об'єктів
    for box in results.boxes:
        cls = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if cls == CLASS_CONTAINER:
            containers.append((x1, y1, x2, y2))
        elif cls == CLASS_STEM:
            stems.append((x1, y1, x2, y2))

    # Малюємо контейнери (сині рамки, товщина 4)
    for (x1, y1, x2, y2) in containers:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)

    # Малюємо стеблини (червоні рамки + номери, товщина 4)
    for idx, (x1, y1, x2, y2) in enumerate(stems, start=1):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(img, str(idx), (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    # Підпис кількості стеблин (правий верхній кут, червоний, збільшений)
    # Виправлено розмір шрифту та позицію для кращого відображення
    text = f"Stems num: {len(stems)}"
    cv2.putText(img, text, (img.shape[1] - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Додаткове вирівнювання для кращого читання
    if len(str(len(stems))) > 2:
        cv2.putText(img, text, (img.shape[1] - 500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Збереження результату
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return len(stems)


# =============================
# Основний цикл обробки
# =============================
experiment_data = {}  # {soil_type: {experiment_name: {date: {group: max_stems}}}}

# Збираємо список усіх фото
all_images = []
for root, dirs, files in os.walk(IMAGES_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(Path(root) / file)

print(f"Знайдено {len(all_images)} зображень для обробки.")

# Обробляємо фото з прогрес-баром
for image_path in tqdm(all_images, desc="Обробка зображень"):
    rel_path = image_path.relative_to(IMAGES_DIR)
    out_path = OUTPUT_DIR / rel_path

    # Визначаємо тип грунту і назву експерименту
    try:
        soil_type = rel_path.parts[0]  # наприклад, "брудний_грунт"
        experiment_name = rel_path.parts[1]  # наприклад, "біонорма_просо"
    except IndexError:
        continue

    group_name = rel_path.parts[-2] if "Група" in rel_path.parts[-2] else None

    if soil_type not in experiment_data:
        experiment_data[soil_type] = {}

    if experiment_name not in experiment_data[soil_type]:
        experiment_data[soil_type][experiment_name] = {}

    # Обробка фото
    stems_count = process_image(image_path, out_path)

    # Дата з імені
    date_str = extract_date_from_filename(image_path.name)
    if not date_str:
        continue

    if date_str not in experiment_data[soil_type][experiment_name]:
        experiment_data[soil_type][experiment_name][date_str] = {}

    if group_name:
        prev = experiment_data[soil_type][experiment_name][date_str].get(group_name, 0)
        experiment_data[soil_type][experiment_name][date_str][group_name] = max(prev, stems_count)


# =============================
# Інтерполяція відсутніх значень у таблиці
# =============================
def interpolate_data(sorted_dates, data_for_dates):
    """Заповнює пропущені дані методом інтерполяції:
    - Якщо значення відсутнє між двома відомими — ставимо середнє.
    - Якщо значення перше чи останнє — ставимо сусіднє.
    """
    groups = ["Група_1", "Група_2", "Група_3"]
    for g in groups:
        values = [data_for_dates[d].get(g, None) for d in sorted_dates]
        # Проходимо по всім датам
        for i in range(len(values)):
            if values[i] is None:
                # Шукаємо ліве відоме
                left = None
                for j in range(i - 1, -1, -1):
                    if values[j] is not None:
                        left = values[j]
                        break

                # Шукаємо праве відоме
                right = None
                for j in range(i + 1, len(values)):
                    if values[j] is not None:
                        right = values[j]
                        break

                if left is not None and right is not None:
                    values[i] = (left + right) // 2
                elif left is not None:
                    values[i] = left
                elif right is not None:
                    values[i] = right

        # Записуємо назад
        for idx, d in enumerate(sorted_dates):
            data_for_dates[d][g] = values[idx]


# =============================
# Створення Excel для кожного експерименту
# =============================
today_str = datetime.today().strftime("%d.%m.%y")
print("Створення Excel-таблиць...")

for soil_type, exps_data in tqdm(experiment_data.items(), desc="Генерація таблиць"):
    for exp_name, dates_data in exps_data.items():
        # Сортуємо дати
        sorted_dates = sorted(dates_data.keys(), key=lambda d: datetime.strptime(d, "%d.%m.%y"))

        # Додавання початкової дати 1.07.25 з нульовими значеннями
        start_date = "01.07.25"
        if start_date not in sorted_dates:
            sorted_dates.insert(0, start_date)
            # Додавання нульових значень для всіх груп на початкову дату
            for group in ["Група_1", "Група_2", "Група_3"]:
                dates_data[start_date] = {group: 0}

        # Інтерполяція даних (з виключенням початкової дати)
        # Створюємо копію для інтерполяції, щоб не змінювати початкові дані
        interpolation_data = {}
        for date in sorted_dates:
            if date == start_date:
                interpolation_data[date] = dates_data[date].copy()
            else:
                interpolation_data[date] = dates_data[date].copy()
        
        # Виконуємо інтерполяцію для всіх дат окрім початкової
        for date in sorted_dates:
            if date == start_date:
                continue  # Пропускаємо початкову дату
        
        interpolate_data(sorted_dates, interpolation_data)

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Results"

        # Заголовок
        ws.append(["Дата", "Група_1", "Група_2", "Група_3"])

        for date_str in sorted_dates:
            row = [date_str]
            if date_str == start_date:
                # Для початкової дати вставляємо 0 для всіх груп (як числа)
                row.extend([0, 0, 0])
            else:
                # Для інших дат використовуємо інтерпольовані значення
                for g in ["Група_1", "Група_2", "Група_3"]:
                    row.append(interpolation_data[date_str].get(g, ""))
            ws.append(row)

        # Створюємо шлях до теки експерименту
        exp_dir = OUTPUT_DIR / soil_type / exp_name
        
        # Переконуємось, що тека існує
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Шлях для збереження Excel файлу у відповідній теці експерименту
        excel_path = exp_dir / f"{exp_name}_{today_str}.xlsx"
        wb.save(excel_path)

print("✅ Готово! Оброблені зображення і таблиці збережено у 'processed_images'.")
