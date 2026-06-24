#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# Ejecución FINAL — DeepLabV3+ (ResNet152) — loss: focal 1.0
# 1 run = 1 VM. Para paralelizar, lanza un script distinto en cada VM:
#     VM #4:  ./run_final_focal.sh
#     VM #5:  ./run_final_ce_focal_lovasz.sh
#     VM #6:  ./run_final_focal_lovasz.sh
# (la arquitectura DeepLabV3+ la fija config.py: DECODER_TYPE="deeplabv3plus")
#
# Uso:
#     conda activate grupo-5
#     tmux new -s final
#     ./run_final_focal.sh
#
# Overrides por variable de entorno (opcional):
#     DATA=/ruta/coco EPOCHS=50 PROJECT=final ./run_final_focal.sh
# ════════════════════════════════════════════════════════════════════════════
set +e
set -u

ARCH="dlv3p"
LOSS_NAME="focal1"
DATA="${DATA:-/home/datasets/coco}"
EPOCHS="${EPOCHS:-50}"
PROJECT="${PROJECT:-final}"               # mismo proyecto W&B para las 6 runs → comparables
VM_TAG="${VM_TAG:-${ARCH}_${LOSS_NAME}}"  # aísla logs y checkpoints de esta VM
RUN_NAME="${ARCH}_r152_${LOSS_NAME}"
CKPT="checkpoints/${VM_TAG}"

mkdir -p logs

echo "════════════════════════════════════════════════════════════"
echo "▶ $(date)  $RUN_NAME"
echo "  arch=DeepLabV3+  backbone=resnet152  epochs=$EPOCHS  data=$DATA"
echo "  loss: --focal-weight 1.0  (resto a 0)"
echo "  ckpt=$CKPT  log=logs/${VM_TAG}.log"
echo "════════════════════════════════════════════════════════════"

# Importante: pasar TODOS los pesos explícitos (incl. 0) para anular los
# defaults de config.py (dice=0.5, focal=0.5).
python main.py \
    --data-root "$DATA" --epochs "$EPOCHS" \
    --wandb-project "$PROJECT" --wandb-run-name "$RUN_NAME" --ckpt-dir "$CKPT" \
    --ce-weight 0 --dice-weight 0 --focal-weight 1.0 --lovasz-weight 0 \
    --ohem-ce-weight 0 --weighted-ce-weight 0 \
    2>&1 | tee "logs/${VM_TAG}.log"

echo "✔ $(date)  $RUN_NAME  rc=${PIPESTATUS[0]}"
