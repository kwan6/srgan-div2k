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

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# KONFIGURASI
# =========================
EPOCH_LIST = [1,2,3,4,5]
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
# HELPER TENSOR → IMAGE (numpy HWC float [0,1])
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

    # Load Model: prefer full checkpoint, fallback to G only
    G = Generator(upscale_factor=2).to(device)
    ckpt_full = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    ckpt_g = CHECKPOINT_DIR / f"G_epoch_{epoch}.pth"

    if ckpt_full.exists():
        ckpt = torch.load(ckpt_full, map_location=device)
        if "G_state" in ckpt:
            G.load_state_dict(ckpt["G_state"])
            print(f"Loaded full checkpoint: {ckpt_full.name}")
        else:
            # fallback: maybe user saved dict differently
            try:
                G.load_state_dict(ckpt)
                print(f"Loaded checkpoint (raw) from: {ckpt_full.name}")
            except Exception:
                raise RuntimeError(f"Cannot load G state from {ckpt_full}")
    elif ckpt_g.exists():
        G.load_state_dict(torch.load(ckpt_g, map_location=device))
        print(f"Loaded G-only checkpoint: {ckpt_g.name} (optimizer state not restored)")
    else:
        print(f"Warning: checkpoint for epoch {epoch} not found (checked {ckpt_full.name} and {ckpt_g.name}). Skipping this epoch.")
        continue

    G.eval()

    # lists for metrics (reset per epoch)
    psnr_bic = []
    psnr_sr = []
    ssim_bic = []
    ssim_sr = []

    saved = 0

    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"Eval epoch {epoch}")
        for idx, (lr, hr) in enumerate(loop):

            lr = lr.to(device)
            hr = hr.to(device)

            bicubic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
            sr = G(lr)

            for i in range(lr.size(0)):
                hr_img = tensor_to_img(hr[i])
                sr_img = tensor_to_img(sr[i])
                bic_img = tensor_to_img(bicubic[i])

                # compute per-patch metrics (may produce inf)
                p_bic = peak_signal_noise_ratio(hr_img, bic_img, data_range=1.0)
                p_sr = peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)

                s_bic = structural_similarity(hr_img, bic_img, channel_axis=-1, data_range=1.0)
                s_sr = structural_similarity(hr_img, sr_img, channel_axis=-1, data_range=1.0)

                psnr_bic.append(p_bic)
                psnr_sr.append(p_sr)
                ssim_bic.append(s_bic)
                ssim_sr.append(s_sr)

                # save sample images (first SAVE_SAMPLES only)
                if saved < SAVE_SAMPLES:
                    def save(img, name):
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(img).save(save_epoch_dir / name)

                    gid = idx * BATCH_SIZE + i
                    save(hr_img,      f"{gid:05d}_HR.png")
                    save(bic_img,     f"{gid:05d}_BICUBIC.png")
                    save(sr_img,      f"{gid:05d}_SRGAN.png")
                    saved += 1

    # filter inf/nan before mean; handle all-inf case
    psnr_bic_arr = np.array(psnr_bic)
    psnr_sr_arr  = np.array(psnr_sr)

    finite_bic = psnr_bic_arr[np.isfinite(psnr_bic_arr)]
    finite_sr  = psnr_sr_arr[np.isfinite(psnr_sr_arr)]

    psnr_bic_mean = float(np.mean(finite_bic)) if finite_bic.size > 0 else float('nan')
    psnr_sr_mean  = float(np.mean(finite_sr))  if finite_sr.size  > 0 else float('nan')

    ssim_bic_arr = np.array(ssim_bic)
    ssim_sr_arr  = np.array(ssim_sr)

    finite_s_bic = ssim_bic_arr[np.isfinite(ssim_bic_arr)]
    finite_s_sr  = ssim_sr_arr[np.isfinite(ssim_sr_arr)]

    ssim_bic_mean = float(np.mean(finite_s_bic)) if finite_s_bic.size > 0 else float('nan')
    ssim_sr_mean  = float(np.mean(finite_s_sr))  if finite_s_sr.size > 0 else float('nan')

    print(f"\n📊 HASIL EPOCH {epoch}")
    print(f"PSNR Bicubic : {psnr_bic_mean:.2f}")
    print(f"PSNR SRGAN   : {psnr_sr_mean:.2f}")
    print(f"SSIM Bicubic : {ssim_bic_mean:.4f}")
    print(f"SSIM SRGAN   : {ssim_sr_mean:.4f}")
