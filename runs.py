"""
Registro canónico de los runs de W&B usados en cada gráfica del informe.

Cada gráfica del informe se construye comparando un subconjunto de runs.
En vez de copiar los run_ids por todos lados, los centralizamos aquí con
nombres legibles ("focal_dice", "ohem_k010", etc.).

Cómo rellenar
-------------
1) Lanza tus experimentos en la VM con `python main.py ...` (W&B online).
2) Ve al dashboard del proyecto en wandb.ai (lo configuras con
   Config.WANDB_PROJECT). Cada run tiene un ID en la URL:
   https://wandb.ai/<entity>/<project>/runs/<RUN_ID>
3) Pega ese RUN_ID en el sitio correspondiente de este fichero (o "" si
   no lo has lanzado todavía).
4) Cualquier script que necesite cargar métricas via wandb.Api() puede
   importar de aquí:
       from runs import EXPERIMENTS, get_run_ids
       ids = get_run_ids("loss_comparison")  # → lista de run_ids del bloque

Estructura
----------
EXPERIMENTS es un dict de "bloques" (un bloque por gráfica/comparación).
Cada bloque tiene: { name_humano: { "run_id": "...", "label": "...",
                                     "description": "..." } }
"""
from __future__ import annotations

# Configuración del proyecto en W&B (debe coincidir con Config.WANDB_PROJECT)
WANDB_ENTITY  = "edxnG05"          # tu usuario / equipo en W&B (ajusta si toca)
WANDB_PROJECT = "finetuning"       # debe coincidir con cfg.WANDB_PROJECT


# ═════════════════════════════════════════════════════════════════════════
#  EXPERIMENTOS POR BLOQUE
# ═════════════════════════════════════════════════════════════════════════
EXPERIMENTS: dict[str, dict[str, dict]] = {

    # ─── Bloque 1: comparativa de losses (la gráfica central del informe) ───
    "loss_comparison": {
        "ce":              {"run_id": "", "label": "CE",                 "description": "Cross-Entropy solo"},
        "dice":            {"run_id": "", "label": "Dice",               "description": "Dice solo"},
        "focal_g2":        {"run_id": "", "label": "Focal (γ=2)",        "description": "Focal solo, gamma=2"},
        "focal_g3":        {"run_id": "", "label": "Focal (γ=3)",        "description": "Focal solo, gamma=3 (más agresivo)"},
        "ce_dice":         {"run_id": "", "label": "CE + Dice",          "description": "Combinación clásica 50/50"},
        "focal_dice_50":   {"run_id": "", "label": "Focal + Dice 50/50", "description": "Default actual"},
        "focal_dice_70":   {"run_id": "", "label": "Focal + Dice 70/30", "description": "Más peso a Focal"},
        "ohem_k025":       {"run_id": "", "label": "OHEM-CE (k=0.25)",   "description": "OHEM con default"},
        "ohem_k010":       {"run_id": "", "label": "OHEM-CE (k=0.10)",   "description": "OHEM más selectivo"},
        "weighted_ce":     {"run_id": "", "label": "Weighted-CE auto",   "description": "CE ponderada por frec. inversa"},
        "weighted_dice":   {"run_id": "", "label": "Weighted-CE + Dice", "description": "Combinación con Dice"},
        "focal_lovasz":    {"run_id": "", "label": "Focal + Lovász",     "description": "Combo IoU surrogate"},
        "triple_combo":    {"run_id": "", "label": "Focal+Dice+Lovász",  "description": "Triple combinación"},
    },

    # ─── Bloque 2: comparativa de backbones ────────────────────────────────
    "backbone_comparison": {
        "resnet18":  {"run_id": "", "label": "ResNet18",  "description": "Backbone más pequeño"},
        "resnet34":  {"run_id": "", "label": "ResNet34",  "description": ""},
        "resnet50":  {"run_id": "", "label": "ResNet50",  "description": "Backbone clásico de seg"},
        "resnet101": {"run_id": "", "label": "ResNet101", "description": ""},
        "resnet152": {"run_id": "", "label": "ResNet152", "description": "Backbone más grande"},
    },

    # ─── Bloque 3: estrategia de freezing ──────────────────────────────────
    "freezing_strategy": {
        "all_frozen":     {"run_id": "", "label": "Todo congelado",     "description": "Solo decoder entrenable"},
        "unfreeze_l4":    {"run_id": "", "label": "Layer4 libre",       "description": ""},
        "unfreeze_l34":   {"run_id": "", "label": "Layer3+4 libre",     "description": "Capas semánticas libres"},
        "unfreeze_l234":  {"run_id": "", "label": "Layer2+3+4 libre",   "description": ""},
        "all_unfrozen":   {"run_id": "", "label": "Todo entrenable",    "description": "Sin freezing"},
    },

    # ─── Bloque 4: con vs sin augmentation ─────────────────────────────────
    "augmentation_ablation": {
        "no_aug":     {"run_id": "", "label": "Sin augmentation", "description": "Solo resize + normalize"},
        "basic_aug":  {"run_id": "", "label": "Solo HFlip",       "description": "Mínimo histórico del proyecto"},
        "full_aug":   {"run_id": "", "label": "Full augmentation","description": "scale+crop+color+blur+rot"},
    },

    # ─── Bloque 5: cumulative ablation (la gráfica narrativa) ──────────────
    # Cada paso añade UNA técnica al anterior.
    "cumulative_ablation": {
        "step1_baseline":     {"run_id": "", "label": "Baseline",                "description": "ResNet50 frozen, CE, sin aug"},
        "step2_aug":          {"run_id": "", "label": "+ Augmentation",          "description": "Añadimos scale+crop+color"},
        "step3_unfreeze":     {"run_id": "", "label": "+ Descongelar L3+L4",     "description": "Capas semánticas libres"},
        "step4_loss_combo":   {"run_id": "", "label": "+ Loss combinada",        "description": "Mejor combinación del bloque 1"},
        "step5_label_smooth": {"run_id": "", "label": "+ Label smoothing",       "description": "0.05"},
        "step6_dropout":      {"run_id": "", "label": "+ Decoder dropout",       "description": "0.1"},
        "step7_ema":          {"run_id": "", "label": "+ EMA",                   "description": "decay 0.9999"},
        # step8 (TTA) NO es un run separado: se mide post-hoc con evaluate.py --tta
        # sobre el checkpoint de step7. Se añade como columna en la tabla del informe.
    },

    # ─── Bloque 6: sweep de Focal gamma ────────────────────────────────────
    "focal_gamma_sweep": {
        "g0":  {"run_id": "", "label": "γ=0 (≡ CE)", "description": "Equivale a CE"},
        "g1":  {"run_id": "", "label": "γ=1",        "description": ""},
        "g2":  {"run_id": "", "label": "γ=2",        "description": "Default RetinaNet"},
        "g3":  {"run_id": "", "label": "γ=3",        "description": ""},
        "g5":  {"run_id": "", "label": "γ=5",        "description": "Muy agresivo"},
    },

    # ─── Bloque 7: sweep de OHEM top_k ─────────────────────────────────────
    "ohem_topk_sweep": {
        "k010": {"run_id": "", "label": "k=0.10", "description": "Solo el 10% más duro"},
        "k020": {"run_id": "", "label": "k=0.20", "description": ""},
        "k025": {"run_id": "", "label": "k=0.25", "description": "Default"},
        "k050": {"run_id": "", "label": "k=0.50", "description": ""},
        "k100": {"run_id": "", "label": "k=1.0 (≡ CE)", "description": "Sin selección"},
    },

    # ─── Bloque 8: variabilidad entre seeds (opcional, sube la nota) ──────
    "seed_variability": {
        "seed42":  {"run_id": "", "label": "Seed 42",  "description": "Mejor setup, seed default"},
        "seed123": {"run_id": "", "label": "Seed 123", "description": "Mejor setup, otro seed"},
        "seed7":   {"run_id": "", "label": "Seed 7",   "description": "Mejor setup, otro seed"},
    },
}


# ═════════════════════════════════════════════════════════════════════════
#  HELPERS DE ACCESO
# ═════════════════════════════════════════════════════════════════════════
def get_run_ids(block: str, only_filled: bool = True) -> list[str]:
    """Devuelve la lista de run_ids de un bloque. Por defecto omite los vacíos."""
    if block not in EXPERIMENTS:
        raise KeyError(f"Bloque desconocido: {block!r}. Disponibles: {list(EXPERIMENTS)}")
    runs = EXPERIMENTS[block]
    return [r["run_id"] for r in runs.values() if (not only_filled) or r["run_id"]]


def get_block(block: str) -> dict[str, dict]:
    """Devuelve el dict completo de un bloque (name → {run_id,label,description})."""
    if block not in EXPERIMENTS:
        raise KeyError(f"Bloque desconocido: {block!r}. Disponibles: {list(EXPERIMENTS)}")
    return EXPERIMENTS[block]


def wandb_url(run_id: str) -> str:
    """Construye la URL pública del run en W&B."""
    return f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}"


def status_report() -> None:
    """Imprime cuántos runs de cada bloque están rellenos vs pendientes."""
    print(f"Proyecto W&B: {WANDB_ENTITY}/{WANDB_PROJECT}\n")
    total_filled = total_missing = 0
    for block, runs in EXPERIMENTS.items():
        filled  = sum(1 for r in runs.values() if r["run_id"])
        missing = len(runs) - filled
        total_filled  += filled
        total_missing += missing
        marker = "✓" if missing == 0 else ("·" if filled == 0 else "~")
        print(f"  {marker} {block:<28s}  {filled}/{len(runs):<3d}  rellenos")
    print(f"\nTotal: {total_filled} rellenos · {total_missing} pendientes")


if __name__ == "__main__":
    status_report()
