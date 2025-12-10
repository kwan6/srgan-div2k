import torch
import numpy as np
from PIL import Image
from pathlib import Path

from src.train.model import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[2]

PATCH_DIR = BASE_DIR / "data" / "DIV2K" / "HR_patches"
SAVE_DIR = BASE_DIR / "outputs" / "reconstructed"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Load model final
model_path = BASE_DIR / "outputs" / "checkpoints" / "G_epoch_20.pth"
G = Generator(upscale_factor=2).to(device)
G.load_state_dict(torch.load(model_path, map_location=device))
G.eval()

# ===============================
# PARAMETER GAMBAR ASLI
# ===============================
ORIG_WIDTH  = 2048   # ganti sesuai gambar DIV2K yang dipilih
ORIG_HEIGHT = 2048

PATCH_SIZE = 128
SCALE = 2
SR_PATCH_SIZE = PATCH_SIZE * SCALE

cols = ORIG_WIDTH // PATCH_SIZE
rows = ORIG_HEIGHT // PATCH_SIZE

canvas = np.zeros((rows * SR_PATCH_SIZE, cols * SR_PATCH_SIZE, 3), dtype=np.uint8)

def to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr).unsqueeze(0).to(device) * 2 - 1

def to_img(t):
    t = (t.squeeze(0).cpu().clamp(-1, 1) + 1) / 2
    t = t.numpy().transpose(1, 2, 0)
    return (t * 255).astype(np.uint8)

patch_files = sorted(PATCH_DIR.glob("*_HR.png"))

idx = 0
for r in range(rows):
    for c in range(cols):
        hr_patch = Image.open(patch_files[idx]).convert("RGB")
        hr_lr = hr_patch.resize((PATCH_SIZE//2, PATCH_SIZE//2), Image.BICUBIC)

        lr_tensor = to_tensor(hr_lr)

        with torch.no_grad():
            sr_tensor = G(lr_tensor)

        sr_img = to_img(sr_tensor)

        y1 = r * SR_PATCH_SIZE
        y2 = y1 + SR_PATCH_SIZE
        x1 = c * SR_PATCH_SIZE
        x2 = x1 + SR_PATCH_SIZE

        canvas[y1:y2, x1:x2] = sr_img

        idx += 1

Image.fromarray(canvas).save(SAVE_DIR / "reconstructed_full_image.png")
print("✅ Gambar utuh berhasil disusun kembali!")
