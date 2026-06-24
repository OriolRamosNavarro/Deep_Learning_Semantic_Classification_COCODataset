"""Quick test: Train on 1 sample for multiple epochs to see if mIoU improves"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from config import Config
from engine import entrenar_una_epoca, validar
from losses import SegmentationLoss
from metrics import SegmentationMetrics
from models.unet import UNet
from transforms import PairedTransform

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load VOC2012
transform = PairedTransform(img_size=cfg.IMG_SIZE, train=True)
voc_train = datasets.VOCSegmentation(
    root="./data", year="2012", image_set="train", download=False, transforms=transform
)

# Overfit on 1 sample
train_ds = Subset(voc_train, [0])
val_ds = train_ds

train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# Create model, loss, optimizer
model = UNet(num_classes=21, pretrained=cfg.PRETRAINED).to(device)
criterion = SegmentationLoss(ignore_index=cfg.IGNORE_INDEX)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # Higher LR for overfit
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
metrics = SegmentationMetrics(num_classes=21, ignore_index=cfg.IGNORE_INDEX)

print("\nTraining on 1 sample for 20 epochs:\n")
print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'Val mIoU':>12}")
print("-" * 45)

for epoch in range(20):
    train_loss = entrenar_una_epoca(model, train_loader, optimizer, criterion, device)
    val_loss, val_metrics = validar(model, val_loader, criterion, metrics, device)
    scheduler.step()
    
    print(f"{epoch:5d} {train_loss:12.4f} {val_loss:12.4f} {val_metrics['mIoU']:12.4f}")
