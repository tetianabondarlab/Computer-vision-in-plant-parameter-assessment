from ultralytics import YOLO

# --- Вибір моделі ---
# Якщо хочете донавчати вже існуючу модель:
# model = YOLO("runs/detect/yolo_stem_focus/weights/last.pt")
model = YOLO("yolov8m.pt")   

# --- Тренування ---
model.train(
    data="data.yaml",          # dataset.yaml
    epochs=300,                # кількість епох
    batch=5,                   # розмір batch
    imgsz=1024,                # розмір зображення
    device=0,                  # GPU (або "cpu")
    optimizer="Adam",         # оптимізатор
    lr0=0.0001,                 # learning rate
    lrf=0.01,                   # кінцевий LR
    augment=True,              # аугментації
    rect=True,                 # прямокутні батчі
    workers=0,
    patience=0,
    name="yolo_m"     # назва експерименту
)

print("✅ Тренування завершено! ")
