import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from tqdm import tqdm
from PIL import Image
import json
from datetime import datetime
import re


from src.train.dataset import load_patch_pairs, SRGANDataset
from src.train.model import Generator


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



# =========================
# KONFIGURASI
# =========================
BATCH_SIZE = 8
SAVE_SAMPLES = 5
SKIP_EXISTING = False  # Set False jika mau re-evaluate yang sudah ada



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
LOGS_DIR = BASE_DIR / "outputs" / "logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)



# =========================
# AUTO DETECT CHECKPOINTS
# =========================
def get_available_epochs():
    """Otomatis detect epoch yang tersedia dari checkpoint files"""
    epochs = set()
    
    if CHECKPOINT_DIR.exists():
        # Cari file G_epoch_*.pth atau checkpoint_epoch_*.pth
        for file in CHECKPOINT_DIR.glob("*.pth"):
            # Extract epoch number dari filename
            match = re.search(r'epoch_(\d+)', file.name)
            if match:
                epoch_num = int(match.group(1))
                epochs.add(epoch_num)
    
    return sorted(list(epochs))


def is_epoch_evaluated(epoch):
    """Check apakah epoch sudah pernah dievaluasi"""
    epoch_dir = SAVE_DIR / f"epoch_{epoch}"
    
    # Check apakah folder ada dan ada file hasil evaluasi
    if epoch_dir.exists():
        sample_files = list(epoch_dir.glob("*_SRGAN.png"))
        if len(sample_files) > 0:
            return True
    
    return False


# Detect available epochs
EPOCH_LIST = get_available_epochs()

if not EPOCH_LIST:
    print("[ERROR] No checkpoint files found in", CHECKPOINT_DIR)
    exit(1)

print(f"\n[INFO] Found {len(EPOCH_LIST)} checkpoint(s): {EPOCH_LIST}")

# Filter out already evaluated epochs
if SKIP_EXISTING:
    original_count = len(EPOCH_LIST)
    EPOCH_LIST = [e for e in EPOCH_LIST if not is_epoch_evaluated(e)]
    skipped_count = original_count - len(EPOCH_LIST)
    
    if skipped_count > 0:
        print(f"[INFO] Skipping {skipped_count} already evaluated epoch(s)")
    
    if not EPOCH_LIST:
        print("[INFO] All epochs already evaluated. Set SKIP_EXISTING=False to re-evaluate.")
        exit(0)

print(f"[INFO] Will evaluate {len(EPOCH_LIST)} epoch(s): {EPOCH_LIST}\n")



# =========================
# LOAD TEST DATA
# =========================
splits = load_patch_pairs()
test_lr, test_hr = splits["test"]


test_dataset = SRGANDataset(test_lr, test_hr)


print("Test samples:", len(test_dataset))



# =========================
# HELPER TENSOR → IMAGE (numpy HWC float [0,1])
# =========================
def tensor_to_img(t):
    t = (t.clamp(-1, 1) + 1) / 2
    t = t.cpu().numpy()
    t = np.transpose(t, (1, 2, 0))
    return t


# =========================
# HELPER UNTUK LOAD CHECKPOINT DENGAN ERROR HANDLING
# =========================
def load_checkpoint_safe(checkpoint_path, device):
    """Load checkpoint dengan error handling dan diagnostics"""
    try:
        # Cek ukuran file
        file_size = os.path.getsize(checkpoint_path)
        print(f"  File size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1000:
            print(f"  [WARNING] File terlalu kecil, kemungkinan corrupt")
            return None
        
        # Load dengan weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint
        
    except Exception as e:
        print(f"  [ERROR] Loading checkpoint: {type(e).__name__}: {str(e)}")
        return None



# =========================
# PREPARE LOG FILES
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_txt_path = LOGS_DIR / f"evaluation_results_{timestamp}.txt"
log_json_path = LOGS_DIR / f"evaluation_results_{timestamp}.json"

# Dictionary untuk menyimpan semua hasil
all_results = {
    "timestamp": timestamp,
    "device": device,
    "test_samples": len(test_dataset),
    "batch_size": BATCH_SIZE,
    "epochs": {}
}

# Buka file txt untuk logging
log_file = open(log_txt_path, "w")
log_file.write("=" * 70 + "\n")
log_file.write("SRGAN EVALUATION RESULTS\n")
log_file.write("=" * 70 + "\n")
log_file.write(f"Timestamp: {timestamp}\n")
log_file.write(f"Device: {device}\n")
log_file.write(f"Test samples: {len(test_dataset)}\n")
log_file.write(f"Batch size: {BATCH_SIZE}\n")
log_file.write(f"Epochs to evaluate: {EPOCH_LIST}\n")
log_file.write("=" * 70 + "\n\n")



# =========================
# LOOP EVALUASI PER EPOCH
# =========================
for epoch in EPOCH_LIST:


    print(f"\n===== Evaluasi Epoch {epoch} =====")
    log_file.write(f"\n{'='*70}\n")
    log_file.write(f"EPOCH {epoch}\n")
    log_file.write(f"{'='*70}\n")


    save_epoch_dir = SAVE_DIR / f"epoch_{epoch}"
    os.makedirs(save_epoch_dir, exist_ok=True)


    # RE-INSTANTIATE MODEL SETIAP EPOCH
    print("Creating new Generator instance...")
    G = Generator(upscale_factor=2).to(device)
    
    ckpt_full = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    ckpt_g = CHECKPOINT_DIR / f"G_epoch_{epoch}.pth"
    
    checkpoint_loaded = False
    checkpoint_info = ""


    # Coba load full checkpoint dulu
    if ckpt_full.exists():
        print(f"Trying full checkpoint: {ckpt_full.name}")
        ckpt = load_checkpoint_safe(ckpt_full, device)
        
        if ckpt is not None:
            if "G_state" in ckpt:
                G.load_state_dict(ckpt["G_state"])
                print(f"[OK] Loaded full checkpoint: {ckpt_full.name}")
                checkpoint_info = f"full checkpoint: {ckpt_full.name}"
                checkpoint_loaded = True
            elif isinstance(ckpt, dict):
                try:
                    G.load_state_dict(ckpt)
                    print(f"[OK] Loaded checkpoint (raw) from: {ckpt_full.name}")
                    checkpoint_info = f"raw checkpoint: {ckpt_full.name}"
                    checkpoint_loaded = True
                except Exception as e:
                    print(f"  [ERROR] Cannot load G state: {e}")
    
    # Jika full checkpoint gagal, coba G-only checkpoint
    if not checkpoint_loaded and ckpt_g.exists():
        print(f"Trying G-only checkpoint: {ckpt_g.name}")
        ckpt = load_checkpoint_safe(ckpt_g, device)
        
        if ckpt is not None:
            try:
                G.load_state_dict(ckpt)
                print(f"[OK] Loaded G-only checkpoint: {ckpt_g.name}")
                checkpoint_info = f"G-only checkpoint: {ckpt_g.name}"
                checkpoint_loaded = True
            except Exception as e:
                print(f"  [ERROR] Cannot load G state: {e}")
    
    # Skip epoch jika tidak ada checkpoint yang berhasil di-load
    if not checkpoint_loaded:
        msg = f"[WARNING] No valid checkpoint found for epoch {epoch}. Skipping."
        print(msg)
        log_file.write(msg + "\n")
        log_file.write(f"Checked: {ckpt_full.name} and {ckpt_g.name}\n")
        
        all_results["epochs"][epoch] = {
            "status": "skipped",
            "reason": "no valid checkpoint"
        }
        continue


    log_file.write(f"Checkpoint loaded: {checkpoint_info}\n")
    
    G.eval()
    
    # VERIFY MODEL WEIGHTS
    first_layer_weight = list(G.parameters())[0].data.mean().item()
    print(f"[DEBUG] First layer weight mean: {first_layer_weight:.6f}")
    log_file.write(f"First layer weight mean: {first_layer_weight:.6f}\n\n")


    # Create new test loader setiap epoch
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


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


                # compute per-patch metrics
                p_bic = peak_signal_noise_ratio(hr_img, bic_img, data_range=1.0)
                p_sr = peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)


                s_bic = structural_similarity(hr_img, bic_img, channel_axis=-1, data_range=1.0)
                s_sr = structural_similarity(hr_img, sr_img, channel_axis=-1, data_range=1.0)


                psnr_bic.append(p_bic)
                psnr_sr.append(p_sr)
                ssim_bic.append(s_bic)
                ssim_sr.append(s_sr)


                # save sample images
                if saved < SAVE_SAMPLES:
                    def save(img, name):
                        img = (img * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(img).save(save_epoch_dir / name)


                    gid = idx * BATCH_SIZE + i
                    save(hr_img,      f"{gid:05d}_HR.png")
                    save(bic_img,     f"{gid:05d}_BICUBIC.png")
                    save(sr_img,      f"{gid:05d}_SRGAN.png")
                    saved += 1


    # filter inf/nan before mean
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


    # Print ke console
    print(f"\n[HASIL EPOCH {epoch}]")
    print(f"PSNR Bicubic : {psnr_bic_mean:.2f}")
    print(f"PSNR SRGAN   : {psnr_sr_mean:.2f}")
    print(f"SSIM Bicubic : {ssim_bic_mean:.4f}")
    print(f"SSIM SRGAN   : {ssim_sr_mean:.4f}")
    
    # Write ke log file
    log_file.write("RESULTS:\n")
    log_file.write(f"  PSNR Bicubic : {psnr_bic_mean:.2f} dB\n")
    log_file.write(f"  PSNR SRGAN   : {psnr_sr_mean:.2f} dB\n")
    log_file.write(f"  SSIM Bicubic : {ssim_bic_mean:.4f}\n")
    log_file.write(f"  SSIM SRGAN   : {ssim_sr_mean:.4f}\n")
    log_file.write(f"  Total samples: {len(finite_sr)}\n")
    log_file.write(f"  Samples saved: {saved}\n")
    log_file.write("\n")
    
    # Simpan ke dictionary untuk JSON
    all_results["epochs"][epoch] = {
        "status": "success",
        "checkpoint": checkpoint_info,
        "first_layer_weight_mean": first_layer_weight,
        "metrics": {
            "psnr_bicubic": round(psnr_bic_mean, 2),
            "psnr_srgan": round(psnr_sr_mean, 2),
            "ssim_bicubic": round(ssim_bic_mean, 4),
            "ssim_srgan": round(ssim_sr_mean, 4)
        },
        "total_samples": len(finite_sr),
        "samples_saved": saved
    }
    
    # Free memory
    del G
    torch.cuda.empty_cache()


# =========================
# FINALIZE LOGS
# =========================
log_file.write("\n" + "=" * 70 + "\n")
log_file.write("EVALUATION COMPLETED\n")
log_file.write("=" * 70 + "\n")
log_file.close()

# Save JSON
with open(log_json_path, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\n[LOG] Results saved to:")
print(f"  - {log_txt_path}")
print(f"  - {log_json_path}")
