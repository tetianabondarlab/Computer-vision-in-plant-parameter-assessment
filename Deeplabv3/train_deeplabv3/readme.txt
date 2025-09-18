train_deeplab.py - скрипт для тренування моделі deeplabv3 на тренувальному наборі з теки "train". кращі ваги мережі зберігаються у файлі "deeplabv3_best.pth". Програма дозволяє довчити модель, якщо цей файл вже є у теці і почати з нуля, якщо його немає. 
infer_deeplab.py - запускає навчену модель, беручі ваги з "deeplabv3_best.pth", на зображення з теки "image.jpg". В результаті зображення сегментується на класи, закладені у процесі тренування.

"train_deeplab.py" – a script for training the DeepLabv3 model on the training dataset located in the "train" folder. The best network weights are saved in the "deeplabv3_best.pth" file. The program allows you to continue training the model if this file already exists in the folder, or start from scratch if it does not.
infer_deeplab.py – runs the trained model using the weights from "deeplabv3_best.pth" on the image "image.jpg". As a result, the image is segmented into the classes defined during training.

Середа розробки: Thonny 4.1.7(Python 3.10.11)
