"""
make_plots.py — gráficas para el informe que W&B no hace bien.

W&B ya da training curves, bar charts de mIoU por loss, IoU por clase, etc.
desde el dashboard sin escribir código (Workspace → Add panel). Este script
solo genera lo que hace falta FUERA de W&B:

  confusion_matrix   Heatmap N×N leído del best.pt (la CM se guarda dentro
                     del checkpoint desde la última versión de main.py).
                     Permite normalizar por fila/columna y filtrar a las
                     top-K clases más frecuentes (útil en COCO con 81 cls).

  runs_table         CSV con best_mIoU + config principal de cada run de un
                     bloque de runs.py. Listo para pegar en el informe.

  cumulative         Bar chart vertical de la cumulative ablation del informe
                     (lee los runs del bloque "cumulative_ablation" desde W&B
                     y anota la ganancia incremental sobre cada barra).

Uso:
  python make_plots.py confusion_matrix --ckpt checkpoints/best.pt \
      --out docs/cm_row.png --normalize row --top-k 20
  python make_plots.py runs_table --block loss_comparison \
      --out docs/loss_table.csv
  python make_plots.py cumulative --out docs/cumulative.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from runs import EXPERIMENTS, WANDB_ENTITY, WANDB_PROJECT


# ════════════════════════════════════════════════════════════════════════
#  1. Confusion matrix heatmap (lee del checkpoint, NO de W&B)
# ════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(ckpt_path: str, out_path: str,
                          normalize: str | None = "row",
                          top_k: int | None = None,
                          select: str = "top") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "confusion_matrix" not in ckpt:
        sys.exit(f"[error] {ckpt_path} no contiene 'confusion_matrix'. "
                 "Reentrena con la versión actual de main.py para que lo guarde.")
    cm = ckpt["confusion_matrix"].float()
    class_names = ckpt.get("class_names") or [f"c{i}" for i in range(cm.shape[0])]

    if normalize == "row":
        cm_n = (cm / cm.sum(dim=1, keepdim=True).clamp(min=1)).numpy()
        title_extra = " (norm. por fila — recall)"
    elif normalize == "col":
        cm_n = (cm / cm.sum(dim=0, keepdim=True).clamp(min=1)).numpy()
        title_extra = " (norm. por columna — precision)"
    elif normalize == "all":
        cm_n = (cm / cm.sum().clamp(min=1)).numpy()
        title_extra = " (norm. al total)"
    else:
        cm_n = cm.numpy()
        title_extra = " (counts crudos)"

    # En COCO (81 clases) la CM es ilegible entera → filtra a K clases.
    #   select="top"   → las K con más píxeles en GT (visión general)
    #   select="worst" → las K presentes con peor IoU (dónde falla el modelo)
    sel_extra = ""
    if top_k and top_k < cm.shape[0]:
        if select == "worst":
            diag    = cm.diag()
            denom   = cm.sum(1) + cm.sum(0) - diag
            iou     = torch.where(denom > 0, diag / denom.clamp(min=1),
                                  torch.zeros_like(diag))
            present = cm.sum(1) > 0                            # solo clases en el GT
            order   = [i for i in iou.argsort().tolist() if present[i]][:top_k]
            sel_extra = f"  ·  peores-{top_k} por IoU"
        else:                                                 # "top" (frecuencia)
            order = cm.sum(dim=1).argsort(descending=True)[:top_k].tolist()
            sel_extra = f"  ·  top-{top_k} por frecuencia"
        sel = sorted(order)
        cm_n        = cm_n[np.ix_(sel, sel)]
        class_names = [class_names[i] for i in sel]

    n = cm_n.shape[0]
    side = max(8, n * 0.30)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm_n, cmap="Blues", aspect="equal",
                   vmin=0, vmax=float(cm_n.max()) if cm_n.max() > 0 else 1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Ground truth")
    ep   = ckpt.get("epoch", "?")
    miou = ckpt.get("mIoU", float("nan"))
    ax.set_title(f"Confusion matrix{title_extra}{sel_extra}  ·  epoch={ep}, mIoU={miou:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Anotar valores si la matriz es manejable
    if n <= 25 and normalize is not None:
        thresh = 0.5
        for i in range(n):
            for j in range(n):
                v = cm_n[i, j]
                if v > 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6,
                            color=("white" if v > thresh else "black"))

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_plots] confusion matrix → {out_path}  ({n}×{n} clases)")


# ════════════════════════════════════════════════════════════════════════
#  2. Runs table CSV (lee de W&B)
# ════════════════════════════════════════════════════════════════════════
def make_runs_table(block: str, out_path: str) -> None:
    try:
        import wandb
    except ImportError:
        sys.exit("[error] wandb no instalado")

    if block not in EXPERIMENTS:
        sys.exit(f"[error] bloque desconocido: {block!r}. "
                 f"Disponibles: {list(EXPERIMENTS)}")

    api = wandb.Api()
    rows = []
    for short_name, info in EXPERIMENTS[block].items():
        run_id = info["run_id"]
        if not run_id:
            print(f"[skip] {short_name}: run_id vacío")
            continue
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        except Exception as e:
            print(f"[warn] {short_name}: no se pudo cargar el run ({e})")
            continue
        cfg = dict(run.config or {})
        sm  = dict(run.summary or {})
        rows.append({
            "name":         short_name,
            "label":        info["label"],
            "best_mIoU":    _fmt(sm.get("best_mIoU")),
            "val_mIoU":     _fmt(sm.get("val_mIoU")),
            "val_loss":     _fmt(sm.get("val_loss")),
            "ce_w":         cfg.get("CE_WEIGHT",          ""),
            "dice_w":       cfg.get("DICE_WEIGHT",        ""),
            "focal_w":      cfg.get("FOCAL_WEIGHT",       ""),
            "lovasz_w":     cfg.get("LOVASZ_WEIGHT",      ""),
            "ohem_w":       cfg.get("OHEM_CE_WEIGHT",     ""),
            "wce_w":        cfg.get("WEIGHTED_CE_WEIGHT", ""),
            "focal_gamma":  cfg.get("FOCAL_GAMMA",        ""),
            "ohem_topk":    cfg.get("OHEM_TOP_K",         ""),
            "backbone":     cfg.get("BACKBONE",           ""),
            "epochs":       cfg.get("EPOCHS",             ""),
            "url":          f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/runs/{run_id}",
        })

    if not rows:
        sys.exit(f"[error] no se ha cargado ningún run del bloque {block!r}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[make_plots] tabla CSV → {out_path}  ({len(rows)} runs)")


def _fmt(v) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


# ════════════════════════════════════════════════════════════════════════
#  3. Cumulative ablation bar chart (lee de W&B + runs.py)
# ════════════════════════════════════════════════════════════════════════
def plot_cumulative(out_path: str) -> None:
    try:
        import wandb
    except ImportError:
        sys.exit("[error] wandb no instalado")

    block = "cumulative_ablation"
    if block not in EXPERIMENTS:
        sys.exit(f"[error] bloque {block!r} no existe en runs.py")

    api = wandb.Api()
    labels, mious = [], []
    for short_name, info in EXPERIMENTS[block].items():
        if not info["run_id"]:
            print(f"[skip] {short_name}: run_id vacío")
            continue
        try:
            run  = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{info['run_id']}")
            miou = run.summary.get("best_mIoU")
            if miou is None:
                print(f"[warn] {short_name}: sin best_mIoU en summary")
                continue
            labels.append(info["label"])
            mious.append(float(miou))
        except Exception as e:
            print(f"[warn] {short_name}: {e}")

    if len(labels) < 2:
        sys.exit(f"[error] insuficientes runs cargados ({len(labels)}). "
                 f"Rellena al menos 2 entradas en EXPERIMENTS['{block}'] de runs.py")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5.5))
    bars = ax.bar(range(len(labels)), mious,
                  color="steelblue", edgecolor="black", linewidth=0.8)

    for i, (bar, m) in enumerate(zip(bars, mious)):
        # Valor absoluto encima
        ax.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.3f}",
                ha="center", va="bottom", fontsize=9)
        # Delta vs anterior dentro de la barra
        if i > 0:
            delta = mious[i] - mious[i - 1]
            sign  = "+" if delta >= 0 else ""
            color = "darkgreen" if delta >= 0 else "darkred"
            ax.text(bar.get_x() + bar.get_width() / 2, m / 2,
                    f"{sign}{delta:+.3f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Best mIoU")
    ax.set_title("Cumulative ablation — contribución de cada técnica")
    ax.set_ylim(0, max(mious) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_plots] cumulative ablation → {out_path}  ({len(labels)} pasos)")


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser(
        description="Plots para el informe (los que W&B no hace bien). "
                    "Para training curves, IoU por clase, comparativas de mIoU "
                    "por run → usa el dashboard de W&B directamente.")
    sub = p.add_subparsers(dest="cmd", required=True)

    cm = sub.add_parser("confusion_matrix",
                        help="Heatmap CM desde un best.pt (la CM va dentro del checkpoint)")
    cm.add_argument("--ckpt",      default="checkpoints/best.pt")
    cm.add_argument("--out",       default="docs/confusion_matrix.png")
    cm.add_argument("--normalize", choices=("none", "row", "col", "all"), default="row",
                    help="row=recall, col=precision, all=fracción del total, none=counts")
    cm.add_argument("--top-k",     type=int, default=None,
                    help="Solo plotea K clases (útil con COCO/81 cls)")
    cm.add_argument("--select",    choices=("top", "worst"), default="top",
                    help="top = K clases con más píxeles en GT; "
                         "worst = K clases presentes con peor IoU")

    rt = sub.add_parser("runs_table",
                        help="CSV con best_mIoU + config de cada run de un bloque")
    rt.add_argument("--block", required=True, choices=list(EXPERIMENTS.keys()))
    rt.add_argument("--out",   default="docs/runs_table.csv")

    cu = sub.add_parser("cumulative",
                        help="Bar chart del bloque cumulative_ablation con "
                             "ganancia incremental por técnica")
    cu.add_argument("--out", default="docs/cumulative_ablation.png")

    args = p.parse_args()

    if args.cmd == "confusion_matrix":
        norm = None if args.normalize == "none" else args.normalize
        plot_confusion_matrix(args.ckpt, args.out, normalize=norm, top_k=args.top_k,
                              select=args.select)
    elif args.cmd == "runs_table":
        make_runs_table(args.block, args.out)
    elif args.cmd == "cumulative":
        plot_cumulative(args.out)


if __name__ == "__main__":
    main()
