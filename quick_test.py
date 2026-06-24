#!/usr/bin/env python3
"""
Sanity check rápido SIN dataset: comprueba que modelo, loss, optimizer, scheduler,
augmentations, AMP y métricas funcionan con tensores aleatorios.

Uso:
    python quick_test.py
"""
import os
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")

import torch

from config import Config
from models.unet import UNet
from losses import SegmentationLoss
from metrics import SegmentationMetrics
from transforms import PairedTransform
from main import construir_optimitzador, build_warmup_cosine_scheduler

print("=" * 70)
print("QUICK TEST — validando el pipeline (sin dataset)")
print("=" * 70)

cfg     = Config()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C, H, W = cfg.NUM_CLASSES, 256, 256
B       = 2
print(f"device={device} | backbone={cfg.BACKBONE} | num_classes={C} | optimizer={cfg.OPTIMIZER}")

# 1) modelo + forward
print("\n[1] modelo + forward...")
model = UNet(num_classes=C, backbone=cfg.BACKBONE, pretrained=False).to(device)
x = torch.randn(B, 3, H, W, device=device)
with torch.no_grad():
    y = model(x)
assert y.shape == (B, C, H, W), f"shape inesperada: {y.shape}"
print(f"    OK — salida {tuple(y.shape)}")

# 2) freeze por capas
print("\n[2] congelación por capas...")
freeze_map = {"layer0": cfg.FREEZE_LAYER0, "layer1": cfg.FREEZE_LAYER1, "layer2": cfg.FREEZE_LAYER2,
              "layer3": cfg.FREEZE_LAYER3, "layer4": cfg.FREEZE_LAYER4}
for ln, fr in freeze_map.items():
    if fr:
        for p in getattr(model.encoder, ln).parameters():
            p.requires_grad = False
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"    OK — {n_tr/1e6:.2f}M params entrenables; congeladas: {[k for k, v in freeze_map.items() if v]}")

# 3) loss
print("\n[3] loss combinada (suma ponderada según cfg)...")
loss_weights = {
    "ce":          cfg.CE_WEIGHT,
    "dice":        cfg.DICE_WEIGHT,
    "focal":       cfg.FOCAL_WEIGHT,
    "lovasz":      cfg.LOVASZ_WEIGHT,
    "ohem_ce":     cfg.OHEM_CE_WEIGHT,
    "weighted_ce": cfg.WEIGHTED_CE_WEIGHT,
}
criterion = SegmentationLoss(
    weights       = loss_weights,
    ignore_index  = cfg.IGNORE_INDEX,
    num_classes   = C,
    focal_gamma   = cfg.FOCAL_GAMMA,
    ohem_top_k    = cfg.OHEM_TOP_K,
    # NOTA: si CLASS_WEIGHTS == "auto" sin cache, el quick_test fallará a propósito
    # (necesitaría un train_loader). Para sanity check pásalo a None o list.
    class_weights = cfg.CLASS_WEIGHTS if cfg.CLASS_WEIGHTS != "auto" else None,
)
print(f"    {criterion!r}")
t = torch.randint(0, C, (B, H, W), device=device)
loss = criterion(y.float(), t)
assert torch.isfinite(loss), "loss no finita"
print(f"    OK — loss = {loss.item():.4f}")

# 4) optimizer + scheduler warmup
print("\n[4] optimizer + scheduler...")
optimizer = construir_optimitzador(model, cfg)
scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=10, total_steps=100)
print(f"    OK — LR inicial dec={optimizer.param_groups[-1]['lr']:.2e}")
for _ in range(5):
    scheduler.step()
print(f"    OK — tras 5 steps de warmup, LR dec={optimizer.param_groups[-1]['lr']:.2e}")

# 5) augmentations
print("\n[5] augmentations sincronizadas...")
from PIL import Image
img  = Image.new("RGB", (320, 320), "red")
mask = Image.new("L",   (320, 320), 3)
it, mt = PairedTransform(img_size=256, train=True)(img, mask)
assert it.shape == (3, 256, 256) and mt.shape == (256, 256), (it.shape, mt.shape)
print(f"    OK — imagen {tuple(it.shape)} | máscara {tuple(mt.shape)}")

# 6) AMP (si hay GPU)
print("\n[6] mixed precision (AMP)...")
if device.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda"):
        out = model(x)
    l = criterion(out.float(), t)
    scaler.scale(l).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"    OK — paso AMP completo, loss = {l.item():.4f}")
else:
    print("    (sin CUDA — se omite)")

# 7) métricas
print("\n[7] métricas (mIoU)...")
metrics = SegmentationMetrics(num_classes=C, ignore_index=cfg.IGNORE_INDEX)
metrics.actualitzar(y, t)
res = metrics.calcular()
print(f"    OK — mIoU = {res['mIoU']:.4f} | {len(res['IoU_per_class'])} IoUs por clase")

print("\n" + "=" * 70)
print("TODO OK — el pipeline está sano. Para entrenar de verdad:")
print(f"  python main.py --data-root /home/datasets/coco --epochs {cfg.EPOCHS}")
print("=" * 70)
