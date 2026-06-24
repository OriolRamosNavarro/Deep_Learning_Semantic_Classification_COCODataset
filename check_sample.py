"""Diagnostic script to check the first VOC2012 sample"""
import torch
from torchvision import datasets
from transforms import PairedTransform
from config import Config

cfg = Config()

# Load the first sample from VOC2012 train set
transform = PairedTransform(img_size=cfg.IMG_SIZE, train=True)
voc_train = datasets.VOCSegmentation(
    root="./data", 
    year="2012", 
    image_set="train", 
    download=False,
    transforms=transform,
)

print(f"Total VOC2012 train samples: {len(voc_train)}")
print()

# Check first few samples
for idx in range(min(5, len(voc_train))):
    image, mask = voc_train[idx]
    
    mask_np = mask.numpy()
    unique_values = torch.unique(mask)
    
    print(f"Sample {idx}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Unique mask values: {unique_values.tolist()}")
    print(f"  Contains ignore_index (255): {(mask == 255).sum().item()} pixels")
    print(f"  Contains valid classes (0-20): {((mask >= 0) & (mask < 21)).sum().item()} pixels")
    print(f"  Total pixels: {mask.numel()}")
    
    # Calculate percentage
    valid_pct = 100 * ((mask >= 0) & (mask < 21)).sum().item() / mask.numel()
    ignore_pct = 100 * (mask == 255).sum().item() / mask.numel()
    print(f"  Valid class pixels: {valid_pct:.1f}%")
    print(f"  Ignore pixels: {ignore_pct:.1f}%")
    print()
