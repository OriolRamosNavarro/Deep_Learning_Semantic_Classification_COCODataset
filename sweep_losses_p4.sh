#!/bin/bash
# Sweep parcial 4/5 — VM #4 (6 runs).
# Reparto balanceado para 5 VM (3 Lovász + 3 no-Lovász por VM).
# Mismo proyecto W&B "Losses".
#
# Runs (posiciones del sweep maestro de 31):
#   14. focal + ohem
#   16. ce + dice + focal
#   18. ce + dice + ohem
#   24. dice + lovasz + ohem
#   25. focal + lovasz + ohem
#   26. ce + dice + focal + lovasz
#
# Uso:
#   conda activate grupo-5
#   tmux new -s losses
#   ./sweep_losses_p4.sh

set +e
set -u

DATA="${DATA:-/home/datasets/coco}"
EPOCHS="${EPOCHS:-30}"
PROJECT="${PROJECT:-Losses}"
VM_TAG="${VM_TAG:-p4}"   # subcarpeta para no colisionar con otras VM por NFS
COMMON="--data-root $DATA --epochs $EPOCHS --wandb-project $PROJECT"

mkdir -p logs

flag_for() {
    case "$1" in
        ce)     echo "--ce-weight" ;;
        dice)   echo "--dice-weight" ;;
        focal)  echo "--focal-weight" ;;
        lovasz) echo "--lovasz-weight" ;;
        ohem)   echo "--ohem-ce-weight" ;;
        *)      echo "ERROR: loss desconocida: $1" >&2; exit 1 ;;
    esac
}

ALL_LOSSES="ce dice focal lovasz ohem"

run_combo() {
    local active_set=" $1 "
    local w="$2"
    local name=""
    local args=""
    for loss in $ALL_LOSSES; do
        if [[ "$active_set" == *" $loss "* ]]; then
            args+=" $(flag_for "$loss") $w"
            name+="${loss}${w}_"
        else
            args+=" $(flag_for "$loss") 0"
        fi
    done
    name="${name%_}"
    local logfile="logs/${VM_TAG}_${name}.log"
    local ckpt="checkpoints/${VM_TAG}/${name}"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "▶ $(date +%H:%M:%S)  $name"
    echo "  log:  $logfile"
    echo "  ckpt: $ckpt"
    echo "════════════════════════════════════════════════════════════"
    python main.py $COMMON --wandb-run-name "$name" --ckpt-dir "$ckpt" $args 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[!] $name terminó con código $rc (continúo con el siguiente)"
    fi
}

START=$(date +%s)
echo "════════════════════════════════════════════════════════════"
echo "  Sweep PARCIAL 4/5 — VM #4: 6 runs"
echo "  proyecto W&B: $PROJECT"
echo "  data:         $DATA"
echo "  epochs/run:   $EPOCHS"
echo "  inicio:       $(date)"
echo "════════════════════════════════════════════════════════════"

run_combo "focal ohem"                  "0.5"     # 14
run_combo "ce dice focal"               "0.33"    # 16
run_combo "ce dice ohem"                "0.33"    # 18
run_combo "dice lovasz ohem"            "0.33"    # 24
run_combo "focal lovasz ohem"           "0.33"    # 25
run_combo "ce dice focal lovasz"        "0.25"    # 26

END=$(date +%s)
DUR=$((END-START))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Sweep p4 terminado en $((DUR/3600))h $(( (DUR%3600)/60 ))m"
echo "  proyecto: $PROJECT (6 runs subidos)"
echo "════════════════════════════════════════════════════════════"
