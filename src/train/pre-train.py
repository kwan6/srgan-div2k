import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import shutil
from pathlib import Path


from src.train.dataset import load_patch_pairs, SRGANDataset
from src.train.model import Generator, Discriminator



# =========================
# KONFIGURASI
# =========================
START_EPOCH = 1      
END_EPOCH   = 5


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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

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
criterion_adv = nn.BCEWithLogitsLoss().to(device)

optimizer_G = Adam(G.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer_D = Adam(D.parameters(), lr=1e-4, betas=(0.9, 0.999))



# =========================
# HELPER: ATOMIC SAVE
# =========================
def save_checkpoint_atomic(state_dict, filepath):
    """
    Save checkpoint dengan atomic write untuk mencegah corruption.
    """
    filepath = os.path.abspath(filepath)
    tmp_path = filepath + ".tmp"
    
    try:
        # Save ke temporary file dulu
        torch.save(state_dict, tmp_path, _use_new_zipfile_serialization=True)
        
        # Verify file size
        if os.path.getsize(tmp_path) < 1000:
            raise RuntimeError(f"Checkpoint file too small (likely corrupt): {tmp_path}")
        
        # Atomic rename
        if os.path.exists(filepath):
            os.replace(tmp_path, filepath)
        else:
            shutil.move(tmp_path, filepath)
        
        # Verify final file
        final_size = os.path.getsize(filepath)
        print(f"[SAVED] {os.path.basename(filepath)} ({final_size / (1024*1024):.2f} MB)")
        
    except Exception as e:
        print(f"[ERROR] Failed saving {os.path.basename(filepath)}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        raise



# =========================
# SETUP DIRECTORIES
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
checkpoint_dir = BASE_DIR / "outputs" / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)



# =========================
# RESUME FROM CHECKPOINT
# =========================
if START_EPOCH > 1:
    last = START_EPOCH - 1
    print(f"\n========= RESUME FROM EPOCH {last} =========")
    
    # Coba load full checkpoint dulu
    ckpt_full = checkpoint_dir / f"checkpoint_epoch_{last}.pth"
    
    if ckpt_full.exists():
        try:
            print(f"Loading full checkpoint: {ckpt_full.name}")
            checkpoint = torch.load(ckpt_full, map_location=device, weights_only=False)
            
            # Load model states
            G.load_state_dict(checkpoint['G_state'])
            D.load_state_dict(checkpoint['D_state'])
            
            # Load optimizer states
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])
            
            print(f"[OK] Resumed from epoch {last}")
            print(f"[OK] G, D, and optimizers loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            raise RuntimeError("Cannot resume training without valid checkpoint")
    
    else:
        print(f"[ERROR] Checkpoint not found: {ckpt_full.name}")
        raise FileNotFoundError(f"Cannot resume from epoch {last}, checkpoint missing")
    
    print("RESUME COMPLETE.\n")



# =========================
# TRAINING LOOP
# =========================
lambda_adv = 1e-3

for epoch in range(START_EPOCH, END_EPOCH + 1):
    
    G.train()
    D.train()
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{END_EPOCH}")
    
    for lr, hr in loop:
        
        lr = lr.to(device)
        hr = hr.to(device)
        
        bs = lr.size(0)
        real_labels = torch.ones(bs, device=device)
        fake_labels = torch.zeros(bs, device=device)
        
        # =========================
        # TRAIN D
        # =========================
        sr = G(lr)
        
        out_real = D(hr).view(bs, -1).mean(dim=1)
        out_fake = D(sr.detach()).view(bs, -1).mean(dim=1)
        
        loss_D = criterion_adv(out_real, real_labels) + criterion_adv(out_fake, fake_labels)
        
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        
        # =========================
        # TRAIN G
        # =========================
        sr = G(lr)
        out_sr = D(sr).view(bs, -1).mean(dim=1)
        
        loss_G_content = criterion_content(sr, hr)
        loss_G_adv = criterion_adv(out_sr, real_labels)
        
        loss_G = loss_G_content + lambda_adv * loss_G_adv
        
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        
        loop.set_postfix({
            "G": float(loss_G.item()),
            "D": float(loss_D.item())
        })
    
    # =========================
    # SAVE CHECKPOINT (ONLY FULL)
    # =========================
    print(f"\nSaving checkpoint for epoch {epoch}...")
    
    try:
        full_checkpoint = {
            'epoch': epoch,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'optimizer_G_state': optimizer_G.state_dict(),
            'optimizer_D_state': optimizer_D.state_dict(),
        }
        
        save_checkpoint_atomic(
            full_checkpoint,
            str(checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
        )
        
        print(f"Checkpoint for epoch {epoch} saved successfully.\n")
        
    except Exception as e:
        print(f"[CRITICAL] Failed to save checkpoint for epoch {epoch}: {e}")
        print("Training will continue, but this epoch checkpoint is lost.\n")

print("\n" + "="*70)
print("TRAINING COMPLETED")
print("="*70)

