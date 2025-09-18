"train.py" - використовується для навчання моделі YOLOv8 на задачу детекції об'єктів (контейнерів та стеблин) у зображеннях рослинних експериментів.

Вибирає модель - або завантажує попередньо навчену модель (last.pt), або використовує стандартну yolov8m.pt для кращої точності

Налаштовує параметри тренування:

Використовує датасет з файлу data.yaml

Після завершення тренування зберігає навчену модель у папку runs/detect/yolo_m/weights/.

Ця модель потім використовується у файлі image_processing.py для аналізу нових зображень.


train.py – used for training a YOLOv8 model on the object detection task (containers and stems) in plant experiment images.

Selects the model – either loads a previously trained model (last.pt) or uses the standard yolov8m.pt for higher accuracy.

Configures training parameters:

Uses the dataset specified in data.yaml.

After training, saves the trained model in the folder runs/detect/yolo_m/weights/.

This model is then used in image_processing.py for analyzing new images.