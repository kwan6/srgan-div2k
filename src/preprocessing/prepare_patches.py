from pathlib import Path
from glob import glob
from PIL import Image
from tqdm import tqdm
import os

BASE_DIR = Path(__file__).resolve().parents[2]

HR_DIR = BASE_DIR / "data" / "DIV2K" / "HR"
HR_PATCH_DIR = BASE_DIR / "data" / "DIV2K" / "HR_patches"
LR_PATCH_DIR = BASE_DIR / "data" / "DIV2K" / "LR_patches"

os.makedirs(HR_PATCH_DIR, exist_ok=True)
os.makedirs(LR_PATCH_DIR, exist_ok=True)

scale = 2
hr_patch_size = 128
lr_patch_size = hr_patch_size // scale

hr_paths = glob(str(HR_DIR / "*.png"))
print("Jumlah HR:", len(hr_paths))

patch_id = 0

for hr_path in tqdm(hr_paths):
    img = Image.open(hr_path).convert("RGB")
    w, h = img.size

    for top in range(0, h - hr_patch_size + 1, hr_patch_size):
        for left in range(0, w - hr_patch_size + 1, hr_patch_size):
            box = (left, top, left + hr_patch_size, top + hr_patch_size)
            hr_patch = img.crop(box)

            lr_patch = hr_patch.resize((lr_patch_size, lr_patch_size), Image.BICUBIC)

            hr_patch.save(HR_PATCH_DIR / f"patch_{patch_id:06d}_HR.png")
            lr_patch.save(LR_PATCH_DIR / f"patch_{patch_id:06d}_LR.png")

            patch_id += 1

print("✅ Patch selesai:", patch_id)
