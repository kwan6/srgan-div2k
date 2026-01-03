from pathlib import Path
from glob import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

HR_PATCH_DIR = BASE_DIR / "data" / "DIV2K" / "HR_patches"
LR_PATCH_DIR = BASE_DIR / "data" / "DIV2K" / "LR_patches"

def load_patch_pairs(seed=42):
    hr_paths = sorted(glob(str(HR_PATCH_DIR / "*_HR.png")))
    lr_paths = sorted(glob(str(LR_PATCH_DIR / "*_LR.png")))

    n = len(hr_paths)
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)

    t1 = int(0.8 * n)
    t2 = int(0.9 * n)

    def pick(paths, ids): return [paths[i] for i in ids]

    return {
        "train": (pick(lr_paths, idx[:t1]), pick(hr_paths, idx[:t1])),
        "val":   (pick(lr_paths, idx[t1:t2]), pick(hr_paths, idx[t1:t2])),
        "test":  (pick(lr_paths, idx[t2:]), pick(hr_paths, idx[t2:]))
    }

class SRGANDataset(Dataset):
    def __init__(self, lr, hr):
        self.lr, self.hr = lr, hr
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, i):
        lr = self.to_tensor(Image.open(self.lr[i]).convert("RGB")) * 2 - 1
        hr = self.to_tensor(Image.open(self.hr[i]).convert("RGB")) * 2 - 1
        return lr, hr

    def __len__(self):
        return len(self.lr)
