#!/bin/bash
# Lanzador del entrenamiento con las optimizaciones activas (ver config.py).
# Ajusta DATA_ROOT y EPOCHS según tu caso.

set -e

DATA_ROOT="${1:-/home/datasets/coco}"
EPOCHS="${2:-30}"

echo "=============================================================================="
echo "ENTRENAMIENTO — config.py controla los hiperparámetros"
echo "  data-root : $DATA_ROOT"
echo "  epochs    : $EPOCHS"
echo "  (resto: BACKBONE / IMG_SIZE / BATCH_SIZE / FREEZE_* / loss / AMP / etc. en config.py)"
echo "=============================================================================="

# Descomenta si necesitas activar el entorno conda:
# conda activate grupo-5

python main.py --data-root "$DATA_ROOT" --epochs "$EPOCHS"

echo "=============================================================================="
echo "ENTRENAMIENTO COMPLETADO"
echo "=============================================================================="
