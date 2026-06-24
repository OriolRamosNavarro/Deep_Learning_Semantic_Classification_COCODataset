#!/bin/bash
# Sanity check de TODAS las técnicas implementadas en el proyecto.
# Cada bloque es un proceso Python aislado: si uno peta, sabes cuál.
# Ejecuta con: ./sanity_checks.sh   (o:  DATA=/otra/ruta ./sanity_checks.sh)

set -e
DATA="${DATA:-/home/datasets/coco}"
EPOCHS=2
SAMPLES=5
ARGS_COMUNES="--overfit $SAMPLES --epochs $EPOCHS --no-wandb --data-root $DATA"

# Helper: aplica patches al Config y luego corre main.principal con los args del CLI.
patch_run() {
    local name="$1"
    local patches="$2"
    shift 2
    echo ""
    echo "============================================================"
    echo "  TEST: $name"
    if [ -n "$patches" ]; then
        echo "  patches: $(echo "$patches" | tr -d '\n' | sed 's/  */ /g')"
    fi
    echo "============================================================"
    python -c "
from config import Config
$patches
import main
main.principal(main.analitzar_arguments())
" "$@"
}

# ─── 0) Tests internos de losses.py ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  TEST: losses.py (tests internos de las 6 losses)"
echo "============================================================"
python losses.py

# ─── 1) Baseline: tal cual está config.py ──────────────────────────────────
patch_run "baseline (config tal cual)" "" $ARGS_COMUNES

# ─── 2) Decoder dropout ────────────────────────────────────────────────────
patch_run "decoder_dropout = 0.1" \
    "Config.DECODER_DROPOUT = 0.1" \
    $ARGS_COMUNES

# ─── 3) Label smoothing (con solo CE para que se note) ─────────────────────
patch_run "label_smoothing = 0.05" \
    "Config.LABEL_SMOOTHING = 0.05" \
    $ARGS_COMUNES --ce-weight 1 --dice-weight 0 --focal-weight 0

# ─── 4) EMA ────────────────────────────────────────────────────────────────
patch_run "EMA (decay = 0.99)" \
    "Config.USE_EMA = True
Config.EMA_DECAY = 0.99" \
    $ARGS_COMUNES

# ─── 5) Scheduler poly ─────────────────────────────────────────────────────
patch_run "scheduler = 'poly' (power=0.9)" \
    "Config.SCHEDULER = 'poly'
Config.POLY_POWER = 0.9" \
    $ARGS_COMUNES

# ─── 6) Scheduler step ─────────────────────────────────────────────────────
patch_run "scheduler = 'step' (size=1, gamma=0.5)" \
    "Config.SCHEDULER = 'step'
Config.STEP_SIZE = 1
Config.STEP_GAMMA = 0.5" \
    $ARGS_COMUNES

# ─── 7) Scheduler constant ─────────────────────────────────────────────────
patch_run "scheduler = 'constant'" \
    "Config.SCHEDULER = 'constant'" \
    $ARGS_COMUNES

# ─── 8) Save every N epochs ────────────────────────────────────────────────
patch_run "SAVE_EVERY_N_EPOCHS = 1 (snapshot por epoch)" \
    "Config.SAVE_EVERY_N_EPOCHS = 1" \
    $ARGS_COMUNES

# ─── 9) Métricas extra (pixel acc + F1 + boundary IoU) ─────────────────────
patch_run "métricas extra (pixel_acc + F1 + boundary_IoU)" \
    "Config.LOG_PIXEL_ACCURACY = True
Config.LOG_F1_PER_CLASS = True
Config.LOG_BOUNDARY_IOU = True" \
    $ARGS_COMUNES

# ─── 10) WANDB_PROJECT custom (offline para no necesitar login) ────────────
patch_run "WANDB_PROJECT custom (offline)" \
    "Config.WANDB_PROJECT = 'sanity-check-test'" \
    --overfit $SAMPLES --epochs $EPOCHS --wandb-offline --data-root $DATA

# ─── 11) TODO junto: la combinación más completa ───────────────────────────
patch_run "TODAS las técnicas a la vez" \
    "Config.DECODER_DROPOUT = 0.1
Config.LABEL_SMOOTHING = 0.05
Config.USE_EMA = True
Config.EMA_DECAY = 0.99
Config.SCHEDULER = 'poly'
Config.POLY_POWER = 0.9
Config.SAVE_EVERY_N_EPOCHS = 1
Config.LOG_PIXEL_ACCURACY = True
Config.LOG_F1_PER_CLASS = True
Config.LOG_BOUNDARY_IOU = True" \
    $ARGS_COMUNES

# ─── 12) TTA en evaluate.py (solo si hay checkpoint del paso anterior) ─────
if [ -f "checkpoints/best.pt" ]; then
    echo ""
    echo "============================================================"
    echo "  TEST: TTA en evaluate.py (checkpoint del test 11)"
    echo "============================================================"
    python evaluate.py \
        --ckpt checkpoints/best.pt \
        --data-root "$DATA" \
        --tta \
        --num-samples 2 \
        --no-figure
else
    echo ""
    echo "[WARN] checkpoints/best.pt no existe; saltando test TTA"
fi

echo ""
echo "============================================================"
echo "  ✓ TODOS LOS SANITY CHECKS PASARON"
echo "============================================================"
