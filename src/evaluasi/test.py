import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.train.dataset import load_patch_pairs, SRGANDataset
from src.train.model import Generator


# =========================
# KONFIGURASI
# =========================
EPOCH_LIST = [5, 10, 15, 20]  # sesuai pembagian anggota
BATCH_SIZE = 8
SAVE_SAMPLES = 10


# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================
# PATH
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
SAVE_DIR = BASE_DIR / "outputs" / "test_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# LOAD TEST DATA
# =========================
splits = load_patch_pairs()
test_lr, test_hr = splits["test"]

test_dataset = SRGANDataset(test_lr, test_hr)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Test batches:", len(test_loader))


# =========================
# HELPER TENSOR → IMAGE
# =========================
def tensor_to_img(t):
    t = (t.clamp(-1, 1) + 1) / 2
    t = t.cpu().numpy()
    t = np.transpose(t, (1, 2, 0))
    return t


# =========================
# LOOP EVALUASI PER EPOCH
# =========================
for epoch in EPOCH_LIST:

    print(f"\n===== Evaluasi Epoch {epoch} =====")

    save_epoch_dir = SAVE_DIR / f"epoch_{epoch}"
    os.makedirs(save_epoch_dir, exist_ok=True)

    # Load Model
    G = Generator(upscale_factor=2).to(device)
    G.load_state_dict(
        torch.load(CHECKPOINT_DIR / f"G_epoch_{epoch}.pth", map_location=device)
    )
    G.eval()

    psnr_bic, psnr_sr = [], []
    ssim_bic, ssim_sr = [], []

    saved = 0

    with torch.no_grad():
        for idx, (lr, hr) in enumerate(tqdm(test_loader)):

            lr = lr.to(device)
            hr = hr.to(device)

            bicubic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
            sr = G(lr)

            for i in range(lr.size(0)):
                hr_img = tensor_to_img(hr[i])
                sr_img = tensor_to_img(sr[i])
                bic_img = tensor_to_img(bicubic[i])

                psnr_bic.append(peak_signal_noise_ratio(hr_img, bic_img, data_range=1.0))
                psnr_sr.append(peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0))

                ssim_bic.append(structural_similarity(hr_img, bic_img, channel_axis=-1, data_range=1.0))
                ssim_sr.append(structural_similarity(hr_img, sr_img, channel_axis=-1, data_range=1.0))

                if saved < SAVE_SAMPLES:
                    def save(img, name):
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(img).save(save_epoch_dir / name)

                    gid = idx * BATCH_SIZE + i
                    save(hr_img,      f"{gid:05d}_HR.png")
                    save(bic_img,     f"{gid:05d}_BICUBIC.png")
                    save(sr_img,      f"{gid:05d}_SRGAN.png")
                    saved += 1

    print(f"\n📊 HASIL EPOCH {epoch}")
    print(f"PSNR Bicubic : {np.mean(psnr_bic):.2f}")
    print(f"PSNR SRGAN   : {np.mean(psnr_sr):.2f}")
    print(f"SSIM Bicubic : {np.mean(ssim_bic):.4f}")
    print(f"SSIM SRGAN   : {np.mean(ssim_sr):.4f}")
