image_processing.py — скрипт для обробки зображень рослин, використовуючи нейронну мережу DeepLabV3 + ResNet50 для сегментації. 

Він:

1. Завантажує модель для розпізнавання різних частин зображення (фон, контейнер, рослина, суха рослина, ґрунт, стебло).
2. Обробляє зображення: 
   - Змінює розмір і додає падінг.
   - Застосовує модель для отримання маски сегментації.
   - Видаляє падінг, обчислює площу зеленої та сухої рослин, висоту рослини.
3. Створює оверлеї на оригінальних зображеннях з кольоровими областями для рослин і додає підписи з обчисленими параметрами.
4. Формує Excel-таблиці:
   - Групує дані за датами та групами.
   - Обчислює середні значення площі і висоти.
5. Зберігає оброблені зображення та Excel-файли у відповідних папках.

Обчислення параметрів рослин:

1. Обчислення площі (в см²)
Спочатку визначається кількість пікселів для кожної класу (зелена рослина, контейнер)
Використовується контейнер як еталон для визначення масштабу.
см_на_піксель = √(площа контейнера / кількість пікселів контейнера).
Площа рослини = кількість пікселів × (масштаб в см)²

2. Обчислення висоти рослини(в см)
Знаходяться координати Y всіх пікселів рослини.
Висота в пікселях = максимальна координата Y - мінімальна координата Y.
Висота в см = висота в пікселях × масштаб (см_на_піксель).


"image\_processing.py" — a script for processing plant images using the DeepLabV3 + ResNet50 neural network for segmentation.

It:

1. Loads the model for recognizing different parts of the image (background, container, plant, dry plant, soil, stem).
2. Processes images:

   * Resizes and adds padding.
   * Applies the model to obtain the segmentation mask.
   * Removes padding, calculates the area of green and dry plants, and the plant height.
3. Creates overlays on the original images with colored regions for plants and adds labels with the calculated parameters.
4. Generates Excel tables:

   * Groups data by dates and groups.
   * Calculates average values of area and height.
5. Saves the processed images and Excel files in the corresponding folders.

Plant parameter calculations:

1. Area calculation (in cm²)

   * First, the number of pixels is determined for each class (green plant, container).
   * The container is used as a reference for determining the scale.
   * `cm_per_pixel = √(container_area / container_pixel_count)`
   * `Plant area = pixel_count × (cm_per_pixel)²`

2. Plant height calculation (in cm)

   * The Y coordinates of all plant pixels are determined.
   * `Height in pixels = max Y coordinate – min Y coordinate`
   * `Height in cm = height_in_pixels × cm_per_pixel`

