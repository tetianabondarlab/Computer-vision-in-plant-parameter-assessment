import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    deeplabv3_mobilenet_v3_large,
)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# === КОНСТАНТИ ===
CLASSES = ['background', 'container', 'dryplant', 'plant', 'soil', 'stem']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 1500  # такий самий як у навчанні

PALETTE = np.array([
    [0, 0, 0],         # background
    [255, 0, 0],       # container
    [255, 255, 0],     # dryplant
    [0, 255, 0],       # plant
    [139, 69, 19],     # soil
    [0, 255, 255],     # stem
], dtype=np.uint8)

# === ВСТАНОВЛЕННЯ ПРИСТРОЮ ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === ФУНКЦІЯ ВИБОРУ МОДЕЛІ ===
def get_model(model_type="resnet50", num_classes=NUM_CLASSES):
    if model_type == "resnet50":
        return deeplabv3_resnet50(num_classes=num_classes)
    elif model_type == "resnet101":
        return deeplabv3_resnet101(num_classes=num_classes)
    elif model_type == "mobilenet":
        return deeplabv3_mobilenet_v3_large(num_classes=num_classes)
    else:
        raise ValueError(f"Невідома модель: {model_type}")

# === ОБРАНА МОДЕЛЬ ===
MODEL_TYPE = "resnet50"  # можна змінювати: "resnet50", "resnet101", "mobilenet"
model = get_model(MODEL_TYPE, num_classes=NUM_CLASSES)

# === ЗАВАНТАЖЕННЯ ЧЕКПОЙНТА ===
checkpoint = torch.load("deeplabv3_best.pth", map_location=device)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# === ТРАНСФОРМАЦІЯ З ASPECT RATIO + ПАДДІНГ ===
def preprocess_with_padding(image, size=IMG_SIZE):
    original_size = image.size  # (W, H)
    w, h = original_size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top

    padded = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

    return padded, (pad_left, pad_top, pad_right, pad_bottom), original_size

# === ІНФЕРЕНС ===
def infer(image_path):
    image = Image.open(image_path).convert("RGB")
    resized_img, padding, orig_size = preprocess_with_padding(image)

    input_tensor = transforms.ToTensor()(resized_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
        prediction = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Відновлення до пропорцій оригінального зображення (без паддінгу)
    pad_left, pad_top, pad_right, pad_bottom = padding
    unpadded = prediction[pad_top:IMG_SIZE - pad_bottom, pad_left:IMG_SIZE - pad_right]
    unpadded_image = Image.fromarray(unpadded.astype(np.uint8), mode='L')
    unpadded_image = unpadded_image.resize(orig_size, Image.NEAREST)

    result = np.array(unpadded_image)
    color_mask = PALETTE[result]

    # === ВІЗУАЛІЗАЦІЯ ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    ax1.imshow(image)
    ax1.set_title("Оригінал")
    ax1.axis("off")

    ax2.imshow(color_mask)
    ax2.set_title(f"Сегментація ({MODEL_TYPE})")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

# === ЗАПУСК ===
infer("image.jpg")
