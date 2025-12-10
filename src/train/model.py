import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Residual Block
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


# =========================
# Upsample Block (PixelShuffle)
# =========================
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * scale_factor ** 2,
            kernel_size=3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


# =========================
# Generator (SRGAN)
# =========================
class Generator(nn.Module):
    def __init__(self, num_residuals=8, upscale_factor=2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.prelu = nn.PReLU()

        # Residual blocks
        res_blocks = []
        for _ in range(num_residuals):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsampling blocks
        upsample_blocks = []
        num_upsamples = int(torch.log2(torch.tensor(upscale_factor)).item())
        for _ in range(num_upsamples):
            upsample_blocks.append(UpsampleBlock(64, 2))
        self.upsample = nn.Sequential(*upsample_blocks)

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x1 = self.prelu(self.conv1(x))
        x2 = self.res_blocks(x1)
        x2 = self.bn2(self.conv2(x2))

        x = x1 + x2  # skip connection
        x = self.upsample(x)
        x = self.conv3(x)

        return torch.tanh(x)


# =========================
# Discriminator (SRGAN)
# =========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            block(256, 512, 1),
            block(512, 512, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
