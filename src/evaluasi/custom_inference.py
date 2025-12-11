import torch
from PIL import Image
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from src.train.model import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[2]
CUSTOM_DIR = BASE_DIR / "data" / "custom"
SAVE_DIR = BASE_DIR / "outputs" / "custom_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load model (default epoch 20)
checkpoint = BASE_DIR / "outputs" / "checkpoints" / "G_epoch_5.pth"
G = Generator(upscale_factor=2).to(device)
G.load_state_dict(torch.load(checkpoint, map_location=device))
G.eval()

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr).unsqueeze(0).to(device) * 2 - 1

def to_img(tensor):
    img = (tensor.squeeze(0).detach().cpu().clamp(-1,1) + 1) / 2
    img = img.numpy().transpose(1,2,0)
    return Image.fromarray((img*255).astype(np.uint8))

# Loop semua foto custom
for img_path in CUSTOM_DIR.glob("*"):
    print("Proses:", img_path.name)

    hr_img = load_image(img_path)

    # Buat LR otomatis (downscale)
    w, h = hr_img.size
    lr_img = hr_img.resize((w//2, h//2), Image.BICUBIC)

    lr_tensor = to_tensor(lr_img)

    with torch.no_grad():
        sr_tensor = G(lr_tensor)

    bicubic_img = lr_img.resize((w, h), Image.BICUBIC)
    sr_img = to_img(sr_tensor)

    bicubic_img.save(SAVE_DIR / f"{img_path.stem}_bicubic.png")
    sr_img.save(SAVE_DIR / f"{img_path.stem}_srgan.png")
    hr_img.save(SAVE_DIR / f"{img_path.stem}_original.png")

print("Selesai! Hasil disimpan di:", SAVE_DIR)
