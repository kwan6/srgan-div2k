import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from src.train.dataset import load_patch_pairs, SRGANDataset
from src.train.model import Generator, Discriminator



# =========================
# KONFIGURASI KELOMPOK (EDIT SESUAI ANGGOTA)
# =========================
START_EPOCH = 2     # ⬅️ ganti sesuai anggota
END_EPOCH   = 5     # ⬅️ ganti sesuai anggota

# Contoh pembagian:
# Orang 1:  1–5
# Orang 2:  6–10
# Orang 3: 11–15
# Orang 4: 16–20


# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================
# LOAD DATASET
# =========================
splits = load_patch_pairs()

train_lr, train_hr = splits["train"]
val_lr, val_hr     = splits["val"]

train_dataset = SRGANDataset(train_lr, train_hr)
val_dataset   = SRGANDataset(val_lr, val_hr)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,     # penting untuk Windows
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Train batches:", len(train_loader))
print("Val batches  :", len(val_loader))


# =========================
# MODEL
# =========================
G = Generator(upscale_factor=2).to(device)
D = Discriminator().to(device)


# =========================
# LOSS & OPTIMIZER
# =========================
criterion_content = nn.L1Loss().to(device)
criterion_adv = nn.BCELoss().to(device)

optimizer_G = Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer_D = Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999))


# =========================
# CHECKPOINT
# =========================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
checkpoint_dir = os.path.join(BASE_DIR, "outputs", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Resume otomatis jika START_EPOCH > 1
if START_EPOCH > 1:
    last = START_EPOCH - 1
    print(f"🔄 Resume dari epoch {last}")

    G.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, f"G_epoch_{last}.pth"),
        map_location=device
    ))
    D.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, f"D_epoch_{last}.pth"),
        map_location=device
    ))


# =========================
# TRAINING CONFIG
# =========================
lambda_adv = 1e-3


# =========================
# TRAINING LOOP
# =========================
for epoch in range(START_EPOCH, END_EPOCH + 1):

    G.train()
    D.train()

    loop = tqdm(train_loader, desc=f"Epoch [{epoch}]")

    for lr, hr in loop:

        lr = lr.to(device)
        hr = hr.to(device)

        batch_size = lr.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # =========================
        # Train Discriminator
        # =========================
        with torch.no_grad():
            sr = G(lr)

        real_out = D(hr)
        fake_out = D(sr.detach())

        loss_D_real = criterion_adv(real_out, real_labels)
        loss_D_fake = criterion_adv(fake_out, fake_labels)
        loss_D = loss_D_real + loss_D_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # =========================
        # Train Generator
        # =========================
        sr = G(lr)
        fake_out = D(sr)

        loss_G_adv = criterion_adv(fake_out, real_labels)
        loss_G_content = criterion_content(sr, hr)

        loss_G = loss_G_content + lambda_adv * loss_G_adv

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        loop.set_postfix(
            loss_G=float(loss_G.item()),
            loss_D=float(loss_D.item())
        )

    # =========================
    # SAVE CHECKPOINT
    # =========================
    torch.save(
        G.state_dict(),
        os.path.join(checkpoint_dir, f"G_epoch_{epoch}.pth")
    )
    torch.save(
        D.state_dict(),
        os.path.join(checkpoint_dir, f"D_epoch_{epoch}.pth")
    )

    print(f"✅ Epoch {epoch} selesai & model disimpan.")
