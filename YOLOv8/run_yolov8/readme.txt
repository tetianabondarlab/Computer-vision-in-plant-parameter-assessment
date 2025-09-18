"image_processing.py" - автоматично обробляє зображення з рослинними експериментами, використовуючи модель YOLOv8 для визначення контейнерів та стеблин.

1. Завантажує модель YOLOv8 для детекції об'єктів на зображеннях.
2. Обробляє фото з теки "images", малюючи рамки навколо контейнерів (сині) та стеблин (червоні), а також нумеруючи стеблин.
3. Рахує кількість стеблин на кожному зображенні.
4. Зберігає оброблені зображення у теку "processed_images".
5. Створює Excel-таблиці для кожного експерименту з даними про кількість стеблин за датами.
6. Використовує дати з імен файлів для групування результатів.


Here’s the English translation:

"image\_processing.py" – automatically processes images from plant experiments using the YOLOv8 model to detect containers and stems.

1. Loads the YOLOv8 model for object detection in images.
2. Processes photos from the "images" folder, drawing bounding boxes around containers (blue) and stems (red), and numbering the stems.
3. Counts the number of stems in each image.
4. Saves the processed images in the *"processed\_images"* folder.
5. Generates Excel tables for each experiment with data on the number of stems by date.
6. Uses dates from file names to group the results.
