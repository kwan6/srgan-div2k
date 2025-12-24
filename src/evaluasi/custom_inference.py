import torch
from PIL import Image
import numpy as np
from pathlib import Path
import torch.nn.functional as F


from src.train.model import Generator


device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = Path(__file__).resolve().parents[2]
CUSTOM_DIR = BASE_DIR / "data" / "custom"
SAVE_DIR = BASE_DIR / "outputs" / "custom_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Load model dari full checkpoint
checkpoint_path = BASE_DIR / "outputs" / "checkpoints" / "checkpoint_epoch_10.pth"

if not checkpoint_path.exists():
    print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
    print("Available checkpoints:")
    for ckpt in (BASE_DIR / "outputs" / "checkpoints").glob("checkpoint_epoch_*.pth"):
        print(f"  - {ckpt.name}")
    exit(1)

G = Generator(upscale_factor=2).to(device)

# Load dari full checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
G.load_state_dict(checkpoint['G_state'])
G.eval()

print(f"[OK] Loaded model from epoch {checkpoint['epoch']}")


def load_image(path):
    img = Image.open(path).convert("RGB")
    return img


def to_tensor(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr).unsqueeze(0).to(device) * 2 - 1


def to_img(tensor):
    img = (tensor.squeeze(0).detach().cpu().clamp(-1,1) + 1) / 2
    img = img.numpy().transpose(1,2,0)
    return Image.fromarray((img*255).astype(np.uint8))


print("\n" + "="*70)
print("SRGAN CUSTOM IMAGE SUPER-RESOLUTION")
print("="*70)
print("\nIMPORTANT NOTES:")
print("- SRGAN hanya untuk upscaling (memperbesar) resolusi")
print("- Input foto HARUS sudah tajam dan clean")
print("- TIDAK bisa memperbaiki: blur, noise, overexposure, atau JPEG artifacts")
print("- Untuk hasil terbaik, gunakan foto berkualitas tinggi yang diresize kecil")
print("="*70 + "\n")


# Loop semua foto custom
image_files = list(CUSTOM_DIR.glob("*.jpg")) + list(CUSTOM_DIR.glob("*.png"))

if not image_files:
    print(f"[ERROR] No images found in {CUSTOM_DIR}")
    print("Supported formats: .jpg, .png")
    exit(1)

print(f"Found {len(image_files)} image(s) to process\n")

for img_path in image_files:
    print(f"Processing: {img_path.name}")
    
    try:
        hr_img = load_image(img_path)
        w, h = hr_img.size
        
        print(f"  Original size: {w}x{h}")
        
        # Buat LR otomatis (downscale)
        lr_img = hr_img.resize((w//2, h//2), Image.BICUBIC)
        print(f"  Downscaled to: {w//2}x{h//2}")
        
        lr_tensor = to_tensor(lr_img)
        
        with torch.no_grad():
            sr_tensor = G(lr_tensor)
        
        bicubic_img = lr_img.resize((w, h), Image.BICUBIC)
        sr_img = to_img(sr_tensor)
        
        # Save results
        lr_img.save(SAVE_DIR / f"{img_path.stem}_1_input_lowres.png")
        bicubic_img.save(SAVE_DIR / f"{img_path.stem}_2_bicubic.png")
        sr_img.save(SAVE_DIR / f"{img_path.stem}_3_srgan.png")
        hr_img.save(SAVE_DIR / f"{img_path.stem}_4_original.png")
        
        print(f"  [SAVED] Results saved to {SAVE_DIR}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to process {img_path.name}: {e}")
    
    print()

print("="*70)
print("PROCESSING COMPLETED")
print(f"Results saved in: {SAVE_DIR}")
print("="*70)
