#!/bin/bash
# Sweep de experimentos para dejar corriendo toda la noche.
#
# Cada bloque lanza un python main.py con configuración distinta y un
# --wandb-run-name claro, así en Wandb cada run queda etiquetado y se
# pueden comparar entre ellos. Todos los runs van al mismo proyecto
# para que aparezcan agrupados en el dashboard.
#
# Uso:
#     ./run_experiments.sh
#     DATA=/otra/ruta ./run_experiments.sh
#     EPOCHS=20 ./run_experiments.sh
#     PROJECT="sweep-loss-2026-05-15" ./run_experiments.sh
#
# Si un experimento falla (OOM, etc.) los siguientes siguen corriendo
# (set +e). Los logs por experimento se guardan en logs/<run_name>.log.
# Para parar todo:  Ctrl-C   (o:  kill %1 %2 ...  desde otra shell)

set +e   # no abortar el sweep entero si un experimento falla
set -u

# ── parámetros comunes ────────────────────────────────────────────────────
DATA="${DATA:-/home/datasets/coco}"
EPOCHS="${EPOCHS:-30}"
PROJECT="${PROJECT:-overnight-sweep-$(date +%Y%m%d)}"
COMMON="--data-root $DATA --epochs $EPOCHS --wandb-project $PROJECT"

mkdir -p logs

run_exp () {
    local name="$1"; shift
    local logfile="logs/${name}.log"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "▶ $(date +%H:%M:%S)  $name"
    echo "  log: $logfile"
    echo "════════════════════════════════════════════════════════════"
    # tee a stdout + log para que se vea progreso en vivo y quede el log
    python main.py $COMMON --wandb-run-name "$name" "$@" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[!] $name terminó con código $rc (continúo con el siguiente)"
    fi
}

START=$(date +%s)
echo "════════════════════════════════════════════════════════════"
echo "  Sweep de experimentos"
echo "  proyecto W&B: $PROJECT"
echo "  data:         $DATA"
echo "  epochs:       $EPOCHS por experimento"
echo "  inicio:       $(date)"
echo "════════════════════════════════════════════════════════════"

# ╭─────────────────────────────────────────────────────────────────────╮
# │ Lista de experimentos. Edítala según lo que queráis comparar:       │
# │                                                                     │
# │   run_exp <nombre>  <flags adicionales de main.py>                  │
# │                                                                     │
# │ Los pesos de la loss son los más útiles para sweepear, junto con    │
# │ --focal-gamma y --ohem-top-k. Para hyperparams sin flag (LR, freeze,│
# │ scheduler, EMA, etc.) toca tocar config.py entre tandas o añadir    │
# │ flags nuevos a main.py.                                             │
# ╰─────────────────────────────────────────────────────────────────────╯

# ── losses: comparar combinaciones ───────────────────────────────────────
run_exp "baseline_focal_dice"      # config tal cual: 0.5 focal + 0.5 dice
run_exp "only_focal"               --focal-weight 1   --dice-weight 0
run_exp "only_dice"                --focal-weight 0   --dice-weight 1
run_exp "ce_dice"                  --ce-weight 0.5    --focal-weight 0  --dice-weight 0.5
run_exp "focal_lovasz"             --focal-weight 0.5 --lovasz-weight 0.5 --dice-weight 0
run_exp "lovasz_dice"              --lovasz-weight 0.5 --dice-weight 0.5 --focal-weight 0
run_exp "ohem_only"                --focal-weight 0   --dice-weight 0   --ohem-ce-weight 1
run_exp "weighted_ce_auto"         --focal-weight 0   --dice-weight 0   --weighted-ce-weight 1   --class-weights auto

# ── focal con gamma alternativo ──────────────────────────────────────────
run_exp "focal_dice_gamma3"        --focal-weight 0.5 --dice-weight 0.5 --focal-gamma 3
run_exp "focal_dice_gamma1"        --focal-weight 0.5 --dice-weight 0.5 --focal-gamma 1

# ── ohem con top-k alternativo ───────────────────────────────────────────
run_exp "ohem_topk0.5"             --focal-weight 0   --dice-weight 0   --ohem-ce-weight 1   --ohem-top-k 0.5

END=$(date +%s)
DUR=$((END-START))
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Sweep terminado en $((DUR/3600))h $(( (DUR%3600)/60 ))m"
echo "  proyecto: $PROJECT"
echo "  abre Wandb para comparar los runs lado a lado"
echo "════════════════════════════════════════════════════════════"
