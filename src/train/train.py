import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import load_patch_pairs, SRGANDataset
from model import Generator, Discriminator
import os

device = "cuda"
splits = load_patch_pairs()

train_lr, train_hr = splits["train"]
train_loader = DataLoader(SRGANDataset(train_lr, train_hr), batch_size=8, shuffle=True, num_workers=0)

G = Generator().to(device)
D = Discriminator().to(device)

loss_L1 = nn.L1Loss()
loss_BCE = nn.BCELoss()

opt_G = Adam(G.parameters(), lr=1e-4)
opt_D = Adam(D.parameters(), lr=1e-4)

epochs = 10
os.makedirs("../../outputs/checkpoints", exist_ok=True)

for e in range(epochs):
    for lr, hr in train_loader:
        lr, hr = lr.to(device), hr.to(device)

        real = torch.ones(lr.size(0), 1).to(device)
        fake = torch.zeros(lr.size(0), 1).to(device)

        # Train D
        sr = G(lr).detach()
        loss_D = loss_BCE(D(hr), real) + loss_BCE(D(sr), fake)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train G
        sr = G(lr)
        loss_G = loss_L1(sr, hr) + 1e-3 * loss_BCE(D(sr), real)
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    torch.save(G.state_dict(), f"../../outputs/checkpoints/G_epoch_{e+1}.pth")
    print(f"✅ Epoch {e+1} selesai")
