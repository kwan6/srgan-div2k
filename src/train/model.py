"""
model.py

Isi:
- ResidualBlock
- UpsampleBlock (PixelShuffle)
- Generator (SRGAN-style)
- Discriminator (PatchGAN-like)
- helper: weights_init

Kompatibel untuk upscale_factor=2 (LR 64x64 -> HR 128x128).
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic building blocks
# -----------------------------
class ResidualBlock(nn.Module):
    """
    Residual block used in SRGAN generator.
    Conv -> BN -> PReLU -> Conv -> BN -> skip
    """
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return identity + out


class UpsampleBlock(nn.Module):
    """
    Upsample block using PixelShuffle.
    Conv (channels -> channels * r^2) -> PixelShuffle(r) -> PReLU
    """
    def __init__(self, in_channels: int, up_scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), kernel_size=3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.ps(x)
        x = self.prelu(x)
        return x


# -----------------------------
# Generator
# -----------------------------
class Generator(nn.Module):
    """
    Generator kompatibel dengan checkpoint lama:
    memakai self.prelu (BUKAN prelu1).
    """
    def __init__(self, in_channels=3, out_channels=3, num_res_blocks=8, base_channels=64, upscale_factor=2):
        super().__init__()

        # Inisial conv
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()  

        # Residual blocks
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(base_channels))
        self.res_blocks = nn.Sequential(*blocks)

        # Setelah residual
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)

        # Upsampling
        assert upscale_factor in [1,2,4]
        ups = []
        if upscale_factor >= 2:
            ups.append(UpsampleBlock(base_channels, up_scale=2))
        if upscale_factor == 4:
            ups.append(UpsampleBlock(base_channels, up_scale=2))
        self.upsample = nn.Sequential(*ups) if len(ups)>0 else nn.Identity()

        # Output
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()

        self._initialize()

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))   # ← PRELU lama
        x2 = self.res_blocks(x1)
        x2 = self.bn2(self.conv2(x2))
        x2 = x2 + x1
        x2 = self.upsample(x2)
        x2 = self.conv3(x2)
        return self.tanh(x2)  # output [-1,1]

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)



# -----------------------------
# Discriminator (PatchGAN-like)
# -----------------------------
class Discriminator(nn.Module):
    """
    Discriminator network similar to SRGAN paper (a sequence of convs with increasing features).
    Produces a single-channel score map (or final dense decision).
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=1, batch_norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(base_channels, base_channels, stride=2, batch_norm=True),   # 64 -> 64
            conv_block(base_channels, base_channels*2, stride=1, batch_norm=True), # 64 -> 128
            conv_block(base_channels*2, base_channels*2, stride=2, batch_norm=True),

            conv_block(base_channels*2, base_channels*4, stride=1, batch_norm=True),
            conv_block(base_channels*4, base_channels*4, stride=2, batch_norm=True),

            conv_block(base_channels*4, base_channels*8, stride=1, batch_norm=True),
            conv_block(base_channels*8, base_channels*8, stride=2, batch_norm=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            
        )

        self._initialize_weights()

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# -----------------------------
# Exported names
# -----------------------------
__all__ = ["Generator", "Discriminator", "ResidualBlock", "UpsampleBlock"]
