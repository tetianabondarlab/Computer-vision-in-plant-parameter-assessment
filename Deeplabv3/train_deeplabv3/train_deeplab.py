import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.models.segmentation as models
from tqdm import tqdm

# === КОНФІГУРАЦІЯ ===
NUM_CLASSES = 6
EPOCHS = 7
BATCH_SIZE = 4
LEARNING_RATE = 1e-6
DATA_DIR = 'train'
IMG_SIZE = 1500
CHECKPOINT_PATH = 'deeplabv3_best.pth'
CONFIG_DEVICE = 'cuda'

# === ВИБІР ПРИСТРОЮ ===
device = torch.device('cuda' if CONFIG_DEVICE == 'cuda' and torch.cuda.is_available() else 'cpu')
print(f"{'✅ GPU' if device.type == 'cuda' else '⚠️ CPU'} використовується")

# === ТРАНСФОРМАЦІЇ З АСПЕКТНИМ СПІВВІДНОШЕННЯМ ===
class ResizeWithAspectRatio:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

class ResizeMaskWithAspectRatio:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.NEAREST)
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2
        pad_right = self.size - new_w - pad_left
        pad_bottom = self.size - new_h - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

transform = transforms.Compose([
    ResizeWithAspectRatio(IMG_SIZE),
    transforms.ToTensor(),
])

mask_transform = transforms.Compose([
    ResizeMaskWithAspectRatio(IMG_SIZE),
    transforms.PILToTensor(),
])

# === DATASET ===
class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [f for f in os.listdir(root_dir)
                       if f.endswith(('.jpg', '.png')) and '_mask' not in f]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + '_mask.png'

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask.long()

# === DATASET & DATALOADER ===
dataset = SegmentationDataset(DATA_DIR, transform=transform, mask_transform=mask_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === МОДЕЛЬ ===
model = models.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
model.to(device)

# === ОПТИМІЗАТОР І ФУНКЦІЯ ВТРАТ ===
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# === BatchNorm Fix ===
def set_bn_eval(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

model.apply(set_bn_eval)

# === ЗАВАНТАЖЕННЯ НАЙКРАЩОЇ МОДЕЛІ (будь-якого формату) ===
best_loss = float('inf')
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"🔁 Завантажено найкращу модель з {CHECKPOINT_PATH} (loss = {best_loss:.4f})")
    else:
        model.load_state_dict(checkpoint)
        print(f"⚠️ Завантажено старий формат вагів із {CHECKPOINT_PATH} (без best_loss)")
else:
    print("🚫 Файл вагів не знайдено. Навчання з нуля.")

# === НАВЧАННЯ ===
for epoch in range(EPOCHS):
    running_loss = 0.0
    model.train()
    model.apply(set_bn_eval)

    for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.to(device), masks.squeeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"📉 Epoch [{epoch+1}/{EPOCHS}] — Loss: {avg_loss:.4f}")

    # === ЗБЕРЕЖЕННЯ НАЙКРАЩОЇ МОДЕЛІ ===
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss
        }, CHECKPOINT_PATH)
        print(f"✅ Збережено нову найкращу модель (loss = {best_loss:.4f})")

# === ЗБЕРЕЖЕННЯ ОСТАННЬОЇ МОДЕЛІ ===
torch.save(model.state_dict(), 'deeplabv3_last.pth')
print("💾 Модель збережено як 'deeplabv3_last.pth'")
