#!/bin/bash
# Sweep parcial: las 10 combinaciones INTERMEDIAS (posiciones 12-21 de 31).
# Para lanzar en una VM distinta a la del sweep_losses.sh principal.
# Todos los runs van al mismo proyecto de W&B ("Losses") con --wandb-run-name
# claro, así aparecen agrupados con los de las otras VM.
#
# Combinaciones (ver sweep_losses.sh para el sweep completo):
#   12. dice + ohem
#   13. focal + lovasz
#   14. focal + ohem
#   15. lovasz + ohem
#   16. ce + dice + focal
#   17. ce + dice + lovasz
#   18. ce + dice + ohem
#   19. ce + focal + lovasz
#   20. ce + focal + ohem
#   21. ce + lovasz + ohem
#
# REQUISITOS en la VM (ver mensaje del chat):
#   1. git pull (rama pauvi con sweep_losses_mid.sh)
#   2. config.py: MASKS_ROOT apuntando a una carpeta tuya con permisos
#   3. python tools/precompute_coco_masks.py ... --split val (~30s)
#      python tools/precompute_coco_masks.py ... --split train (~8 min)
#   4. tmux + ./sweep_losses_mid.sh
#
# Uso:
#   ./sweep_losses_mid.sh
#   DATA=/otra/ruta ./sweep_losses_mid.sh
#   EPOCHS=15 ./sweep_losses_mid.sh
#   PROJECT="Losses" ./sweep_losses_mid.sh

set +e
set -u

DATA="${DATA:-/home/datasets/coco}"
EPOCHS="${EPOCHS:-30}"
PROJECT="${PROJECT:-Losses}"
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
    local logfile="logs/${name}.log"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "▶ $(date +%H:%M:%S)  $name"
    echo "  log: $logfile"
    echo "════════════════════════════════════════════════════════════"
    python main.py $COMMON --wandb-run-name "$name" $args 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[!] $name terminó con código $rc (continúo con el siguiente)"
    fi
}

START=$(date +%s)
echo "════════════════════════════════════════════════════════════"
echo "  Sweep INTERMEDIO (posiciones 12-21 de 31): 10 runs"
echo "  proyecto W&B: $PROJECT"
echo "  data:         $DATA"
echo "  epochs/run:   $EPOCHS"
echo "  inicio:       $(date)"
echo "════════════════════════════════════════════════════════════"

# ── 10 combinaciones (mezcla de k=2 final y todo k=3) ──────────────────
run_combo "dice ohem"          "0.5"      # 12
run_combo "focal lovasz"       "0.5"      # 13
run_combo "focal ohem"         "0.5"      # 14
run_combo "lovasz ohem"        "0.5"      # 15
run_combo "ce dice focal"      "0.33"     # 16
run_combo "ce dice lovasz"     "0.33"     # 17
run_combo "ce dice ohem"       "0.33"     # 18
run_combo "ce focal lovasz"    "0.33"     # 19
run_combo "ce focal ohem"      "0.33"     # 20
run_combo "ce lovasz ohem"     "0.33"     # 21

END=$(date +%s)
DUR=$((END-START))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Sweep intermedio terminado en $((DUR/3600))h $(( (DUR%3600)/60 ))m"
echo "  proyecto: $PROJECT (10 runs subidos)"
echo "════════════════════════════════════════════════════════════"
