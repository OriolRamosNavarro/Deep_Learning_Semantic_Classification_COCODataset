"""Test script to verify metrics calculation"""
import torch
from metrics import SegmentationMetrics

# Test case 1: Perfect predictions
print("=== Test 1: Perfect Predictions ===")
metrics = SegmentationMetrics(num_classes=21, ignore_index=255)

# Create fake predictions: all class 0
batch_size, channels, h, w = 2, 21, 256, 256
preds = torch.zeros(batch_size, channels, h, w)
preds[:, 0, :, :] = 10  # high score for class 0

# Create targets: all class 0
targets = torch.zeros(batch_size, h, w, dtype=torch.long)

metrics.actualitzar(preds, targets)
result = metrics.calcular()
print(f"mIoU: {result['mIoU']:.4f}")
print(f"Expected: 1.0 (all pixels correctly predicted)")
print()

# Test case 2: All ignore_index (255)
print("=== Test 2: All Ignore Index ===")
metrics = SegmentationMetrics(num_classes=21, ignore_index=255)

preds = torch.zeros(batch_size, channels, h, w)
preds[:, 0, :, :] = 10

targets = torch.full((batch_size, h, w), 255, dtype=torch.long)

metrics.actualitzar(preds, targets)
result = metrics.calcular()
print(f"mIoU: {result['mIoU']:.4f}")
print(f"Expected: 0.0 (no valid pixels to evaluate)")
print()

# Test case 3: Mix of valid and ignore pixels
print("=== Test 3: Mix of Valid and Ignore Pixels ===")
metrics = SegmentationMetrics(num_classes=21, ignore_index=255)

preds = torch.zeros(batch_size, channels, h, w)
preds[:, 0, :, :] = 10  # class 0

targets = torch.zeros(batch_size, h, w, dtype=torch.long)
targets[:, :128, :] = 255  # top half is ignore

metrics.actualitzar(preds, targets)
result = metrics.calcular()
print(f"mIoU: {result['mIoU']:.4f}")
print(f"Expected: 1.0 (all non-ignored pixels correctly predicted)")
print()

# Test case 4: Wrong predictions
print("=== Test 4: Wrong Predictions ===")
metrics = SegmentationMetrics(num_classes=21, ignore_index=255)

preds = torch.zeros(batch_size, channels, h, w)
preds[:, 1, :, :] = 10  # predicts class 1

targets = torch.zeros(batch_size, h, w, dtype=torch.long)  # targets are class 0

metrics.actualitzar(preds, targets)
result = metrics.calcular()
print(f"mIoU: {result['mIoU']:.4f}")
print(f"Expected: close to 0.0 (incorrect predictions)")
