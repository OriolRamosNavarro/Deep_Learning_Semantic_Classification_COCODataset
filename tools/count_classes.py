"""Cuenta en cuántas IMÁGENES aparece cada clase de COCO (80 + background).

Sirve para ver el desbalanceo de clases del dataset. Cuenta a nivel de imagen:
"número de imágenes que contienen ≥1 instancia (no-crowd) de la clase C". Usa las
anotaciones (no las máscaras), filtrando iscrowd=0 para cuadrar con cómo
tools/precompute_coco_masks.py generó las máscaras.

Mapeo de category_id (1-90, no contiguos) → índice 1..80 (0=fondo) vía
classes.COCO_CAT_ID_TO_INDEX, idéntico al de las máscaras.

Uso (desde la raíz del repo):
    python tools/count_classes.py --coco-root /home/datasets/coco --split train
    python tools/count_classes.py --coco-root /home/datasets/coco --split val
    python tools/count_classes.py --coco-root /home/datasets/coco --split train \
        --csv docs/class_counts_train.csv --plot docs/class_counts_train.png

Notas:
  - "background" aparece en todas las imágenes → se reporta = nº total de imágenes.
  - El recuento por anotaciones puede ser ligeramente mayor que por máscaras: una
    instancia pequeña totalmente ocluida por otra mayor podría no quedar en la
    máscara final (se pintan de mayor a menor área). Para desbalanceo es el estándar.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

# permitir 'from classes import ...' al ejecutar el script desde tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes import COCO_CAT_ID_TO_INDEX, COCO_CLASSES  # noqa: E402

_SPLIT_DIR = {"train": "train2017", "val": "val2017"}


def count_images_per_class(coco_root: str, split: str):
    """Devuelve lista de (idx, nombre, n_imgs) ordenada por idx, + total de imágenes."""
    from pycocotools.coco import COCO

    split_dir = _SPLIT_DIR[split]
    ann_file  = os.path.join(coco_root, "annotations", f"instances_{split_dir}.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"No existe el fichero de anotaciones: {ann_file}")

    coco = COCO(ann_file)
    total_imgs = len(coco.imgs)

    # Una sola pasada por todas las anotaciones: img_ids únicos por category_id.
    imgs_per_cat = defaultdict(set)
    for ann in coco.loadAnns(coco.getAnnIds()):
        if ann.get("iscrowd", 0):
            continue
        imgs_per_cat[ann["category_id"]].add(ann["image_id"])

    rows = []
    for cat_id, idx in sorted(COCO_CAT_ID_TO_INDEX.items(), key=lambda kv: kv[1]):
        rows.append((idx, COCO_CLASSES[idx], len(imgs_per_cat.get(cat_id, ()))))
    return rows, total_imgs


def main() -> None:
    p = argparse.ArgumentParser(description="Cuenta imágenes por clase en COCO (desbalanceo)")
    p.add_argument("--coco-root", required=True, help="Carpeta raíz de COCO (con annotations/)")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--csv",  default=None, help="Si se indica, escribe el recuento a este CSV")
    p.add_argument("--plot", default=None, help="Si se indica, guarda un bar chart PNG ordenado")
    p.add_argument("--sort", choices=["idx", "count"], default="count",
                   help="Orden de la tabla impresa: por índice de clase o por nº de imágenes")
    args = p.parse_args()

    rows, total = count_images_per_class(args.coco_root, args.split)

    # Tabla impresa (sin background; background = total).
    shown = sorted(rows, key=(lambda r: r[2]) if args.sort == "count" else (lambda r: r[0]))
    print(f"\n=== COCO {args.split}: imágenes por clase  (total imágenes = {total}) ===")
    print(f"{'idx':>3}  {'clase':<16} {'n_imgs':>8} {'% imgs':>8}")
    print("-" * 40)
    print(f"{0:>3}  {'background':<16} {total:>8} {100.0:>7.1f}%   (todas)")
    for idx, name, n in shown:
        print(f"{idx:>3}  {name:<16} {n:>8} {100.0 * n / total:>7.1f}%")

    counts = [n for _, _, n in rows]
    cmin, cmax = min(counts), max(counts)
    print("-" * 40)
    print(f"min  = {cmin}  ({COCO_CLASSES[[idx for idx,_,n in rows if n==cmin][0]]})")
    print(f"max  = {cmax}  ({COCO_CLASSES[[idx for idx,_,n in rows if n==cmax][0]]})")
    print(f"ratio max/min = {cmax / max(1, cmin):.1f}x  → indicador de desbalanceo")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "clase", "n_imgs", "pct_imgs"])
            w.writerow([0, "background", total, f"{100.0:.2f}"])
            for idx, name, n in sorted(rows, key=lambda r: r[0]):
                w.writerow([idx, name, n, f"{100.0 * n / total:.2f}"])
        print(f"\n[count_classes] CSV → {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ordered = sorted(rows, key=lambda r: r[2])          # ascendente por nº imgs
        names = [name for _, name, _ in ordered]
        vals  = [n for _, _, n in ordered]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(range(len(vals)), vals, color="steelblue", edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_ylabel("nº de imágenes")
        ax.set_title(f"COCO {args.split} — imágenes por clase (desbalanceo)  ·  "
                     f"total={total}, ratio max/min={cmax / max(1, cmin):.1f}x")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[count_classes] bar chart → {args.plot}")


if __name__ == "__main__":
    main()
