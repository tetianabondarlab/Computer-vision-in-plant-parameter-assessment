import os
from datetime import datetime
from collections import defaultdict

import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd

# === КОНСТАНТИ ===
CLASSES = ['background', 'container', 'dryplant', 'plant', 'soil', 'stem']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 1500
CONTAINER_WIDTH_CM = 16
CONTAINER_HEIGHT_CM = 15

PALETTE = np.array([
    [0, 0, 0],       # background
    [255, 0, 0],     # container
    [255, 255, 0],   # dryplant
    [0, 255, 0],     # plant
    [139, 69, 19],   # soil
    [0, 255, 255],   # stem
], dtype=np.uint8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Завантаження моделі ===
print("[INFO] Завантаження моделі...")
try:
    model = deeplabv3_resnet50(num_classes=NUM_CLASSES)
    checkpoint = torch.load("deeplabv3_best.pth", map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("[INFO] Модель успішно завантажена")
except Exception as e:
    print(f"[ERROR] Помилка завантаження моделі: {e}")
    exit(1)

# === Препроцесинг ===
def preprocess_with_padding(image, size=IMG_SIZE):
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top

    padded = transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
    return padded, (pad_left, pad_top, pad_right, pad_bottom), (w, h)

# === Аналіз одного зображення ===
def analyze_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        padded_img, padding, orig_size = preprocess_with_padding(image)
        input_tensor = transforms.ToTensor()(padded_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)['out']
            pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        # Видаляємо паддінг
        pad_left, pad_top, pad_right, pad_bottom = padding
        pred = pred[pad_top:IMG_SIZE - pad_bottom, pad_left:IMG_SIZE - pad_right]
        pred = Image.fromarray(pred.astype(np.uint8)).resize(orig_size, Image.NEAREST)
        result = np.array(pred)

        plant_id = CLASSES.index('plant')
        dryplant_id = CLASSES.index('dryplant')
        container_id = CLASSES.index('container')

        # Пікселі
        plant_px = np.sum(result == plant_id)
        dry_px = np.sum(result == dryplant_id)
        container_px = np.sum(result == container_id)

        # Перевід у см²
        cm_per_pixel = np.sqrt((CONTAINER_WIDTH_CM * CONTAINER_HEIGHT_CM) / max(container_px, 1))
        plant_area = plant_px * (cm_per_pixel ** 2)
        dry_area = dry_px * (cm_per_pixel ** 2)

        # Висота рослини
        mask = (result == plant_id) | (result == dryplant_id)
        ys, _ = np.where(mask)
        height_cm = 0
        if len(ys) > 0:
            height_px = ys.max() - ys.min()
            height_cm = height_px * cm_per_pixel

        return result, plant_area, dry_area, height_cm
    except Exception as e:
        print(f"[ERROR] Помилка обробки зображення {image_path}: {e}")
        return None, 0, 0, 0

# === Нанесення оверлею та підписів ===
def overlay_and_save(orig_path, result_mask, plant_area, dry_area, height_cm, save_path):
    try:
        orig = Image.open(orig_path).convert("RGB")
        overlay = Image.new("RGBA", orig.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)

        # Накладення масок
        for cls_id, color in zip([CLASSES.index('plant'), CLASSES.index('dryplant')],
                                 [(0,255,0,80),(255,255,0,80)]):
            mask = (result_mask == cls_id)
            if mask.any():
                mask_img = Image.fromarray((mask*255).astype(np.uint8))
                color_img = Image.new("RGBA", orig.size, color)
                overlay.paste(color_img, (0,0), mask_img)

        combined = Image.alpha_composite(orig.convert("RGBA"), overlay)

        # Підписи
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        if font:
            text = f"Green sq.cm.: {plant_area:.2f}\nDry sq.cm.: {dry_area:.2f}\nLength cm: {height_cm:.2f}"
            draw.multiline_text((10,10), text, fill=(255,0,0), font=font)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined.convert("RGB").save(save_path)
    except Exception as e:
        print(f"[ERROR] Помилка збереження зображення {save_path}: {e}")

# === Обробка усіх зображень і створення таблиць ===
def process_images_and_save_tables(root_dir="images", processed_root="processed_images"):
    today = datetime.now().strftime("%Y%m%d")
    
    # Перевірка наявності директорій
    if not os.path.exists(root_dir):
        print(f"[ERROR] Директорія {root_dir} не існує")
        return
    
    for soil_type in os.listdir(root_dir):
        soil_path = os.path.join(root_dir, soil_type)
        if not os.path.isdir(soil_path):
            continue
        for experiment in os.listdir(soil_path):
            exp_path = os.path.join(soil_path, experiment)
            if not os.path.isdir(exp_path):
                continue

            print(f"[INFO] {soil_type}/{experiment}")
            all_data = defaultdict(lambda: defaultdict(list))

            # === Обхід груп ===
            for group in os.listdir(exp_path):
                group_path = os.path.join(exp_path, group)
                if not os.path.isdir(group_path):
                    continue

                save_group_path = os.path.join(processed_root, soil_type, experiment, group)
                os.makedirs(save_group_path, exist_ok=True)

                for file in os.listdir(group_path):
                    if not file.lower().endswith((".jpg", ".png")):
                        continue
                    img_path = os.path.join(group_path, file)

                    # Витяг дати з імені
                    try:
                        date_str = file.split("_")[1][:8]  # IMG_20250716_xxx → 20250716
                        date = datetime.strptime(date_str, "%Y%m%d").strftime("%d.%m.%y")
                    except Exception as e:
                        print(f"    [WARN] Пропуск {file} (немає дати): {e}")
                        continue

                    result_mask, plant_area, dry_area, height_cm = analyze_image(img_path)
                    if result_mask is not None:  # Перевірка успішності обробки
                        save_path = os.path.join(save_group_path, file)
                        overlay_and_save(img_path, result_mask, plant_area, dry_area, height_cm, save_path)

                        all_data[date][group].append((plant_area, dry_area, height_cm))

            # === Формування загальної таблиці для експерименту ===
            if all_data:
                dates = sorted(all_data.keys(), key=lambda d: datetime.strptime(d, "%d.%m.%y"))
                groups = sorted({g for data in all_data.values() for g in data.keys()})

                # Додаємо початкову дату "1.07.25" з нульовими значеннями
                initial_date = "01.07.25"
                if initial_date not in dates:
                    dates.insert(0, initial_date)  # Вставляємо на початок

                # MultiIndex для колонок
                arrays = []
                for metric in ["Площа зеленої", "Площа суха", "Висота"]:
                    for g in groups:
                        arrays.append((metric, g))
                columns = pd.MultiIndex.from_tuples(arrays)

                df = pd.DataFrame(index=dates, columns=columns)

                # Заповнюємо дані для існуючих дат
                for date in dates:
                    if date == initial_date:
                        # Для початкової дати - всі значення 0
                        for g in groups:
                            df.at[date, ("Площа зеленої", g)] = 0.0
                            df.at[date, ("Площа суха", g)] = 0.0
                            df.at[date, ("Висота", g)] = 0.0
                    else:
                        # Для інших дат - заповнюємо реальними значеннями
                        for g in groups:
                            if g in all_data[date]:
                                vals = all_data[date][g]
                                avg_plant = np.mean([v[0] for v in vals])
                                avg_dry = np.mean([v[1] for v in vals])
                                avg_height = np.mean([v[2] for v in vals])
                                df.at[date, ("Площа зеленої", g)] = avg_plant
                                df.at[date, ("Площа суха", g)] = avg_dry
                                df.at[date, ("Висота", g)] = avg_height

                # === Інтерполяція і заповнення країв ===
                df = df.astype(float)
                
                # Перевірка наявності даних перед інтерполяцією
                if not df.empty and df.isnull().any().any():
                    df = df.interpolate(method="linear", axis=0).ffill().bfill()

                # форматування з комою
                df = df.round(2)  # залишаємо числа з 2 знаками після коми

                df.index.name = "Дата"

                excel_name = f"{experiment}_{today}.xlsx"
                excel_path = os.path.join(processed_root, soil_type, experiment, excel_name)
                os.makedirs(os.path.dirname(excel_path), exist_ok=True)
                
                try:
                    df.to_excel(excel_path, merge_cells=True)
                    print(f"[INFO] Таблиця збережена: {excel_path}")
                except Exception as e:
                    print(f"[ERROR] Помилка збереження Excel файлу: {e}")

if __name__ == "__main__":
    try:
        process_images_and_save_tables()
        print("[INFO] Готово! Оброблені зображення та таблиці збережено")
    except Exception as e:
        print(f"[ERROR] Помилка в головній функції: {e}")
