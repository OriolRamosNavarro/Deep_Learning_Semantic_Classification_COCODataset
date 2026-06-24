"""Diagnostic: Check what the model is actually predicting"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from config import Config
from models.unet import UNet
from transforms import PairedTransform

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load first sample
transform = PairedTransform(img_size=cfg.IMG_SIZE, train=True)
voc = datasets.VOCSegmentation(
    root="./data", year="2012", image_set="train", download=False, transforms=transform
)

image, mask = voc[0]
print(f"Sample 0 analysis:")
print(f"  Mask shape: {mask.shape}")
print(f"  Mask unique values: {torch.unique(mask).tolist()}")

# Get class distribution for classes that appear
mask_valid = mask[mask != 255]
unique_classes, counts = torch.unique(mask_valid, return_counts=True)
print(f"\n  Class distribution (valid pixels only):")
for cls, cnt in zip(unique_classes.tolist(), counts.tolist()):
    pct = 100 * cnt / mask_valid.numel()
    print(f"    Class {cls:2d}: {cnt:6d} pixels ({pct:5.1f}%)")

# Baseline: what happens if model predicts majority class everywhere?
majority_class = unique_classes[counts.argmax()].item()
print(f"\n  Majority class: {majority_class}")
print(f"  If model predicts class {majority_class} everywhere:")
print(f"    - Perfect on class {majority_class}")
print(f"    - 0% on other classes")
print(f"    - mIoU = ~{100/len(unique_classes):.1f}% (1/num_present_classes)")
print(f"    - Expected mIoU ≈ {1/len(unique_classes):.4f}")

# Load model and check predictions
print(f"\n  Loading model and checking predictions...")
model = UNet(num_classes=21, pretrained=cfg.PRETRAINED).to(device)

# Try loading a checkpoint if available
try:
    ckpt_path = "./checkpoints/best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"  Loaded checkpoint from {ckpt_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best mIoU at save: {checkpoint.get('mIoU', 'unknown')}")
except Exception as e:
    print(f"  No checkpoint found, using untrained model: {e}")

model.eval()

with torch.no_grad():
    # Make prediction on sample 0
    image_batch = image.unsqueeze(0).to(device)
    preds = model(image_batch)
    pred_classes = preds.argmax(dim=1).squeeze(0).cpu()
    
    print(f"\n  Model predictions:")
    print(f"    Pred shape: {pred_classes.shape}")
    unique_preds, pred_counts = torch.unique(pred_classes, return_counts=True)
    print(f"    Predicted classes: {unique_preds.tolist()}")
    print(f"    Distribution:")
    for cls, cnt in zip(unique_preds.tolist(), pred_counts.tolist()):
        pct = 100 * cnt / pred_classes.numel()
        print(f"      Class {cls:2d}: {pct:5.1f}%")
    
    # Calculate what mIoU would be
    print(f"\n  Manual mIoU calculation:")
    total_iou = 0
    for cls in unique_classes:
        cls = cls.item()
        target_pixels = (mask == cls).sum().item()
        correct_pixels = ((mask == cls) & (pred_classes == cls)).sum().item()
        if target_pixels > 0:
            iou = correct_pixels / target_pixels
            print(f"    Class {cls}: {correct_pixels}/{target_pixels} = IoU {iou:.4f}")
            total_iou += iou
    
    manual_miou = total_iou / len(unique_classes)
    print(f"  Manual mIoU: {manual_miou:.4f}")
