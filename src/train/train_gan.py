import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import shutil
from pathlib import Path

# Import dari file kamu yang sudah ada
from src.train.dataset import load_patch_pairs, SRGANDataset
from src.train.model import Generator, Discriminator

# =========================
# KONFIGURASI LANJUTAN
# =========================
PRETRAIN_CHECKPOINT = "checkpoint_epoch_34.pth" # Nama file terakhir dari pretrain
START_EPOCH_GAN     = 35                        # Lanjut dari epoch ini
END_EPOCH_GAN       = 40                        # Target finish
BATCH_SIZE          = 8
LR_G                = 1e-4
LR_D                = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Mode: SRGAN Training (VGG + Adv)")

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DEFINISI VGG LOSS (Content Loss)
# =========================
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Load VGG19 pretrained
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Ambil layer sampai ke-36 (sebelum aktivasi ReLU terakhir di blok 5)
        # Ini standar paper SRGAN/ESRGAN agar tekstur lebih natural
        self.loss_network = nn.Sequential(*list(vgg)[:36]).eval().to(device)
        
        # Bekukan parameter VGG (kita tidak training VGG-nya)
        for param in self.loss_network.parameters():
            param.requires_grad = False

        # Normalisasi input VGG (Mean & Std ImageNet)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, sr, hr):
        # Input SR/HR range [-1, 1], ubah ke [0, 1] dulu
        sr = (sr + 1) / 2
        hr = (hr + 1) / 2

        # Normalisasi sesuai standar VGG
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        # Hitung MSE antara feature map SR dan HR
        sr_features = self.loss_network(sr)
        hr_features = self.loss_network(hr)
        
        return nn.functional.mse_loss(sr_features, hr_features)

# =========================
# LOAD DATASET
# =========================
splits = load_patch_pairs()
train_lr, train_hr = splits["train"]
val_lr, val_hr     = splits["val"]

train_loader = DataLoader(SRGANDataset(train_lr, train_hr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"Train batches: {len(train_loader)}")

# =========================
# INIT MODEL & OPTIMIZER
# =========================
G = Generator(upscale_factor=2).to(device)
D = Discriminator().to(device)

optimizer_G = Adam(G.parameters(), lr=LR_G, betas=(0.9, 0.999))
optimizer_D = Adam(D.parameters(), lr=LR_D, betas=(0.9, 0.999))

# =========================
# LOSS FUNCTIONS
# =========================
criterion_content = VGGLoss().to(device)      # Untuk Tekstur
criterion_adv     = nn.BCEWithLogitsLoss().to(device) # Untuk Realism

# =========================
# LOAD PRE-TRAINED WEIGHTS (ESTAFET)
# =========================
ckpt_path = CHECKPOINT_DIR / PRETRAIN_CHECKPOINT

if ckpt_path.exists():
    print(f"\n[INFO] Loading Pre-trained Generator from: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Load bobot Generator (Estafet ilmu)
    if 'G_state' in checkpoint:
        G.load_state_dict(checkpoint['G_state'])
    else:
        G.load_state_dict(checkpoint) # Backup jika format save beda
        
    # Discriminator mulai dari nol (atau load jika mau, tapi biasanya fresh start oke)
    print("[OK] Generator weights loaded successfully.")
else:
    print(f"\n[CRITICAL ERROR] File {PRETRAIN_CHECKPOINT} tidak ditemukan!")
    print("Pastikan kamu sudah rename train.py lama dan hasil checkpointnya ada.")
    exit(1)

# =========================
# TRAINING LOOP (GAN PHASE)
# =========================
for epoch in range(START_EPOCH_GAN, END_EPOCH_GAN + 1):
    G.train()
    D.train()
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{END_EPOCH_GAN} [GAN Training]")
    
    loss_g_accum = 0.0
    loss_d_accum = 0.0

    for lr, hr in loop:
        lr = lr.to(device)
        hr = hr.to(device)
        bs = lr.size(0)

        # ---------------------
        # 1. TRAIN DISCRIMINATOR
        # ---------------------
        optimizer_D.zero_grad()
        
        # Generate gambar palsu (Super Resolution)
        sr = G(lr)
        
        # Prediksi D pada gambar Asli (HR) -> Target 1
        pred_real = D(hr)
        loss_d_real = criterion_adv(pred_real, torch.ones_like(pred_real) - 0.05) # Label smoothing
        
        # Prediksi D pada gambar Palsu (SR) -> Target 0
        pred_fake = D(sr.detach()) # Detach agar G tidak kena update di sini
        loss_d_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))
        
        # Total Loss D
        loss_D = (loss_d_real + loss_d_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        # 2. TRAIN GENERATOR
        # ---------------------
        optimizer_G.zero_grad()
        
        # Kita perlu forward ulang atau pakai sr yg tadi (tapi graph harus retain kalau pakai yg tadi)
        # Lebih aman forward ulang untuk graph baru:
        pred_fake_g = D(sr) # Tanpa detach!
        
        # a. Content Loss (VGG) - Agar mirip aslinya
        loss_content = criterion_content(sr, hr)
        
        # b. Adversarial Loss (Agar D mengira ini asli/1)
        loss_adversarial = criterion_adv(pred_fake_g, torch.ones_like(pred_fake_g))
        
        # Total Loss G (Rumus SRGAN: Content + 0.001 * Adversarial)
        loss_G = loss_content + (1e-3 * loss_adversarial)
        
        loss_G.backward()
        optimizer_G.step()
        
        # Logging
        loss_g_accum += loss_G.item()
        loss_d_accum += loss_D.item()
        
        loop.set_postfix({
            "Loss G": f"{loss_G.item():.4f}",
            "Loss D": f"{loss_D.item():.4f}"
        })

    # =========================
    # SAVE CHECKPOINT
    # =========================
    save_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'G_state': G.state_dict(),
        'D_state': D.state_dict(), # Simpan D juga sekarang
        'optimizer_G_state': optimizer_G.state_dict(),
        'optimizer_D_state': optimizer_D.state_dict(),
    }, save_path)
    
    print(f"Saved checkpoint: {save_path.name}")

print("\nTRAINING GAN SELESAI! Silakan cek hasil visual.")