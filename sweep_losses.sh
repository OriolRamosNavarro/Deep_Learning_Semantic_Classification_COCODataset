#!/bin/bash
# Sweep de las 31 combinaciones posibles de 5 losses (CE, Dice, Focal, Lovasz,
# OHEM-CE) sobre la rama actual. Todos los runs se mandan al mismo proyecto de
# W&B ("Losses") con un nombre que refleja qué losses están activas y su peso.
#
# Para cada tamaño de grupo k ∈ {1, 2, 3, 4, 5}, las k losses activas se reparten
# el peso 1 a partes iguales (1/k):
#     k=1 → peso 1.0  por loss
#     k=2 → peso 0.5  por loss
#     k=3 → peso 0.33 por loss
#     k=4 → peso 0.25 por loss
#     k=5 → peso 0.2  por loss
# Las losses NO activas reciben peso 0 explícitamente (importante: si no se las
# desactiva con un flag CLI, los defaults de config.py — dice=0.5, focal=0.5 —
# las sumarían a la mezcla).
#
# Total: C(5,1)+C(5,2)+C(5,3)+C(5,4)+C(5,5) = 5+10+10+5+1 = 31 runs.
#
# Uso:
#     ./sweep_losses.sh
#     DATA=/otra/ruta ./sweep_losses.sh
#     EPOCHS=20 ./sweep_losses.sh                # baja epochs para no tardar tanto
#     PROJECT="Losses-2026-05" ./sweep_losses.sh # otro nombre de proyecto
#
# Si un run peta (OOM, NaN, etc.) los siguientes siguen corriendo (set +e).
# Cada run deja su log en logs/<run_name>.log.
#
# ⚠️ AVISO DE TIEMPO: 31 runs × 50 epochs en COCO con ResNet152 en una L40S =
# *muchísimas* horas (~600 h en el peor caso). Lánzalo en tmux y considera bajar
# EPOCHS=15 ó 20 (suele bastar para distinguir cuáles losses son competitivas).

set +e    # no abortar el sweep entero si un run falla
set -u

# ── parámetros comunes ────────────────────────────────────────────────────
DATA="${DATA:-/home/datasets/coco}"
EPOCHS="${EPOCHS:-30}"
PROJECT="${PROJECT:-Losses}"
COMMON="--data-root $DATA --epochs $EPOCHS --wandb-project $PROJECT"

mkdir -p logs

# Mapeo nombre corto → flag CLI (ohem → --ohem-ce-weight)
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

# run_combo "<losses activas separadas por espacio>" "<peso por loss>"
#   p.ej. run_combo "ce dice" "0.5"
# Construye el nombre del run y los flags. Las losses NO activas se pasan a 0.
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
    name="${name%_}"                          # quita el _ final
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
echo "  Sweep de combinaciones de losses (31 runs)"
echo "  proyecto W&B: $PROJECT"
echo "  data:         $DATA"
echo "  epochs/run:   $EPOCHS"
echo "  inicio:       $(date)"
echo "════════════════════════════════════════════════════════════"

# ╭─────────────────────────────────────────────────────────────────────╮
# │ C(5,1) — 5 combinaciones, peso 1.0 por loss                         │
# ╰─────────────────────────────────────────────────────────────────────╯
run_combo "ce"      "1.0"
run_combo "dice"    "1.0"
run_combo "focal"   "1.0"
run_combo "lovasz"  "1.0"
run_combo "ohem"    "1.0"

# ╭─────────────────────────────────────────────────────────────────────╮
# │ C(5,2) — 10 combinaciones, peso 0.5 por loss                        │
# ╰─────────────────────────────────────────────────────────────────────╯
run_combo "ce dice"      "0.5"
run_combo "ce focal"     "0.5"
run_combo "ce lovasz"    "0.5"
run_combo "ce ohem"      "0.5"
run_combo "dice focal"   "0.5"
run_combo "dice lovasz"  "0.5"
run_combo "dice ohem"    "0.5"
run_combo "focal lovasz" "0.5"
run_combo "focal ohem"   "0.5"
run_combo "lovasz ohem"  "0.5"

# ╭─────────────────────────────────────────────────────────────────────╮
# │ C(5,3) — 10 combinaciones, peso 0.33 por loss                       │
# ╰─────────────────────────────────────────────────────────────────────╯
run_combo "ce dice focal"      "0.33"
run_combo "ce dice lovasz"     "0.33"
run_combo "ce dice ohem"       "0.33"
run_combo "ce focal lovasz"    "0.33"
run_combo "ce focal ohem"      "0.33"
run_combo "ce lovasz ohem"     "0.33"
run_combo "dice focal lovasz"  "0.33"
run_combo "dice focal ohem"    "0.33"
run_combo "dice lovasz ohem"   "0.33"
run_combo "focal lovasz ohem"  "0.33"

# ╭─────────────────────────────────────────────────────────────────────╮
# │ C(5,4) — 5 combinaciones, peso 0.25 por loss                        │
# ╰─────────────────────────────────────────────────────────────────────╯
run_combo "ce dice focal lovasz"     "0.25"
run_combo "ce dice focal ohem"       "0.25"
run_combo "ce dice lovasz ohem"      "0.25"
run_combo "ce focal lovasz ohem"     "0.25"
run_combo "dice focal lovasz ohem"   "0.25"

# ╭─────────────────────────────────────────────────────────────────────╮
# │ C(5,5) — 1 combinación, peso 0.2 por loss                           │
# ╰─────────────────────────────────────────────────────────────────────╯
run_combo "ce dice focal lovasz ohem"  "0.2"

END=$(date +%s)
DUR=$((END-START))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Sweep terminado en $((DUR/3600))h $(( (DUR%3600)/60 ))m"
echo "  proyecto: $PROJECT"
echo "  abre Wandb para comparar los 31 runs lado a lado"
echo "════════════════════════════════════════════════════════════"
