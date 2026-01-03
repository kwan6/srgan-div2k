import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import shutil
from pathlib import Path

# Import module custom
from src.train.dataset import load_patch_pairs, SRGANDataset
from src.train.model import Generator, Discriminator

# =========================
# KONFIGURASI HYPERPARAMETER
# =========================
PRETRAIN_CHECKPOINT = "checkpoint_epoch_34.pth" 
START_EPOCH_GAN     = 25                        
END_EPOCH_GAN       = 40                        
BATCH_SIZE          = 8
LR_G                = 1e-4
LR_D                = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Mode: SRGAN Training (VGG + Adv)")

# =========================
# PENGATURAN PATH
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = BASE_DIR / "outputs" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# VGG LOSS (PERCEPTUAL LOSS)
# =========================
class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Menggunakan VGG19 pretrained sebagai feature extractor
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        # Mengambil layer hingga ke-36 (sebelum aktivasi ReLU blok 5) untuk menangkap fitur tekstur
        self.loss_network = nn.Sequential(*list(vgg)[:36]).eval().to(device)
        
        # Bekukan parameter (VGG hanya untuk inferensi loss)
        for param in self.loss_network.parameters():
            param.requires_grad = False

        # Register buffer normalisasi (ImageNet standard)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, sr, hr):
        # Denormalisasi [-1, 1] ke [0, 1]
        sr = (sr + 1) / 2
        hr = (hr + 1) / 2

        # Normalisasi input sesuai standar VGG
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        # Hitung MSE loss pada feature map
        sr_features = self.loss_network(sr)
        hr_features = self.loss_network(hr)
        
        return nn.functional.mse_loss(sr_features, hr_features)

# =========================
# DATASET & DATALOADER
# =========================
splits = load_patch_pairs()
train_lr, train_hr = splits["train"]

train_loader = DataLoader(SRGANDataset(train_lr, train_hr), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
print(f"Train batches: {len(train_loader)}")

# =========================
# INISIALISASI MODEL
# =========================
G = Generator(upscale_factor=2).to(device)
D = Discriminator().to(device)

optimizer_G = Adam(G.parameters(), lr=LR_G, betas=(0.9, 0.999))
optimizer_D = Adam(D.parameters(), lr=LR_D, betas=(0.9, 0.999))

# =========================
# LOSS FUNCTIONS
# =========================
criterion_content = VGGLoss().to(device)              # Perceptual Loss (Tekstur)
criterion_adv     = nn.BCEWithLogitsLoss().to(device) # Adversarial Loss (Realism)

# =========================
# LOAD WEIGHTS (PRE-TRAINED)
# =========================
ckpt_path = CHECKPOINT_DIR / PRETRAIN_CHECKPOINT

if ckpt_path.exists():
    print(f"\n[INFO] Loading Pre-trained Generator: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Load state dict Generator saja untuk inisialisasi fase GAN
    if 'G_state' in checkpoint:
        G.load_state_dict(checkpoint['G_state'])
    else:
        G.load_state_dict(checkpoint) 
        
    print("[OK] Generator weights loaded.")
else:
    print(f"\n[ERROR] Checkpoint {PRETRAIN_CHECKPOINT} not found.")
    exit(1)

# =========================
# TRAINING LOOP
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

        # ---------------------
        # 1. UPDATE DISCRIMINATOR
        # ---------------------
        optimizer_D.zero_grad()
        
        sr = G(lr)
        
        # Loss D pada data Real (HR) dengan label smoothing (target 0.95)
        pred_real = D(hr)
        loss_d_real = criterion_adv(pred_real, torch.ones_like(pred_real) - 0.05)
        
        # Loss D pada data Fake (SR) dengan target 0
        # Detach digunakan agar gradien tidak mengalir ke Generator saat update D
        pred_fake = D(sr.detach()) 
        loss_d_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_d_real + loss_d_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        # ---------------------
        # 2. UPDATE GENERATOR
        # ---------------------
        optimizer_G.zero_grad()
        
        # Re-compute forward pass untuk graph komputasi G
        pred_fake_g = D(sr)
        
        # a. Content Loss: Mempertahankan struktur visual
        loss_content = criterion_content(sr, hr)
        
        # b. Adversarial Loss: Menipu D agar mengira SR adalah Real (target 1)
        loss_adversarial = criterion_adv(pred_fake_g, torch.ones_like(pred_fake_g))
        
        # Weighted Total Loss (Standard SRGAN: Content + 1e-3 * Adv)
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
    # SAVING CHECKPOINT
    # =========================
    save_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'G_state': G.state_dict(),
        'D_state': D.state_dict(),
        'optimizer_G_state': optimizer_G.state_dict(),
        'optimizer_D_state': optimizer_D.state_dict(),
    }, save_path)
    
    print(f"Saved checkpoint: {save_path.name}")

print("\nGAN Training Completed.")