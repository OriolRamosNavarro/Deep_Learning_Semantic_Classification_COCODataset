#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# Ejecución FINAL — DeepLabV3+ (ResNet152) — loss: focal 0.5 + lovasz 0.5
# 1 run = 1 VM. Ver run_final_focal.sh para el reparto entre VMs.
# (la arquitectura DeepLabV3+ la fija config.py: DECODER_TYPE="deeplabv3plus")
#
# Uso:
#     conda activate grupo-5
#     tmux new -s final
#     ./run_final_focal_lovasz.sh
# ════════════════════════════════════════════════════════════════════════════
set +e
set -u

ARCH="dlv3p"
LOSS_NAME="focal.5_lovasz.5"
DATA="${DATA:-/home/datasets/coco}"
EPOCHS="${EPOCHS:-50}"
PROJECT="${PROJECT:-final}"
VM_TAG="${VM_TAG:-${ARCH}_focal_lovasz}"
RUN_NAME="${ARCH}_r152_${LOSS_NAME}"
CKPT="checkpoints/${VM_TAG}"

mkdir -p logs

echo "════════════════════════════════════════════════════════════"
echo "▶ $(date)  $RUN_NAME"
echo "  arch=DeepLabV3+  backbone=resnet152  epochs=$EPOCHS  data=$DATA"
echo "  loss: --focal-weight 0.5 --lovasz-weight 0.5"
echo "  ckpt=$CKPT  log=logs/${VM_TAG}.log"
echo "════════════════════════════════════════════════════════════"

python main.py \
    --data-root "$DATA" --epochs "$EPOCHS" \
    --wandb-project "$PROJECT" --wandb-run-name "$RUN_NAME" --ckpt-dir "$CKPT" \
    --ce-weight 0 --dice-weight 0 --focal-weight 0.5 --lovasz-weight 0.5 \
    --ohem-ce-weight 0 --weighted-ce-weight 0 \
    2>&1 | tee "logs/${VM_TAG}.log"

echo "✔ $(date)  $RUN_NAME  rc=${PIPESTATUS[0]}"
