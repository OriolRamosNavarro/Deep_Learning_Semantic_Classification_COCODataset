#!/bin/bash
# Sweep parcial: las 10 combinaciones FINALES (posiciones 22-31 de 31).
# Para lanzar en una VM distinta a la del sweep_losses.sh principal.
# Todos los runs van al mismo proyecto de W&B ("Losses") con --wandb-run-name
# claro, así aparecen agrupados con los de las otras VM.
#
# Combinaciones (ver sweep_losses.sh para el sweep completo):
#   22. dice + focal + lovasz
#   23. dice + focal + ohem
#   24. dice + lovasz + ohem
#   25. focal + lovasz + ohem
#   26. ce + dice + focal + lovasz
#   27. ce + dice + focal + ohem
#   28. ce + dice + lovasz + ohem
#   29. ce + focal + lovasz + ohem
#   30. dice + focal + lovasz + ohem
#   31. ce + dice + focal + lovasz + ohem  (todas)
#
# REQUISITOS en la VM (ver mensaje del chat):
#   1. git pull (rama pauvi con sweep_losses_end.sh)
#   2. config.py: MASKS_ROOT apuntando a una carpeta tuya con permisos
#   3. python tools/precompute_coco_masks.py ... --split val (~30s)
#      python tools/precompute_coco_masks.py ... --split train (~8 min)
#   4. tmux + ./sweep_losses_end.sh
#
# OJO: 8 de las 10 incluyen Lovász → esta tanda es la más lenta.
#
# Uso:
#   ./sweep_losses_end.sh
#   DATA=/otra/ruta ./sweep_losses_end.sh
#   EPOCHS=15 ./sweep_losses_end.sh
#   PROJECT="Losses" ./sweep_losses_end.sh

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
echo "  Sweep FINAL (posiciones 22-31 de 31): 10 runs"
echo "  proyecto W&B: $PROJECT"
echo "  data:         $DATA"
echo "  epochs/run:   $EPOCHS"
echo "  inicio:       $(date)"
echo "════════════════════════════════════════════════════════════"

# ── Las 10 combinaciones restantes: 4 de k=3 + 5 de k=4 + 1 de k=5 ─────
run_combo "dice focal lovasz"            "0.33"     # 22
run_combo "dice focal ohem"              "0.33"     # 23
run_combo "dice lovasz ohem"             "0.33"     # 24
run_combo "focal lovasz ohem"            "0.33"     # 25
run_combo "ce dice focal lovasz"         "0.25"     # 26
run_combo "ce dice focal ohem"           "0.25"     # 27
run_combo "ce dice lovasz ohem"          "0.25"     # 28
run_combo "ce focal lovasz ohem"         "0.25"     # 29
run_combo "dice focal lovasz ohem"       "0.25"     # 30
run_combo "ce dice focal lovasz ohem"    "0.2"      # 31

END=$(date +%s)
DUR=$((END-START))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Sweep final terminado en $((DUR/3600))h $(( (DUR%3600)/60 ))m"
echo "  proyecto: $PROJECT (10 runs subidos)"
echo "════════════════════════════════════════════════════════════"
