"""mIoU estratificado por TAMAÑO de objeto (estándar de áreas de COCO).

Buckets COCO (área en píxeles de la máscara GT, resolución original):
    small  : área < 32²        (< 1024 px)
    medium : 32² ≤ área < 96²  (1024–9216 px)
    large  : área ≥ 96²        (≥ 9216 px)

Como esto es segmentación SEMÁNTICA (no de instancia), la noción "IoU por instancia"
se define así, por cada instancia GT de COCO val:
  - clase c = categoría de la instancia (remapeada 1..80 con COCO_CAT_ID_TO_INDEX).
  - bucket  = según ann["area"] (área de la máscara original, igual que COCO).
  - máscara GT M_i = pycocotools.annToMask(ann)  (binaria, resolución original).
  - predicción localizada P_i = (pred_semántica == c) recortada al bbox de la
    instancia (para no mezclar instancias lejanas de la misma clase).
  - IoU_i = |M_i ∩ P_i| / |M_i ∪ P_i|.
La predicción del modelo (256×256) se reescala a la resolución original (nearest)
antes de comparar, para que las áreas COCO (en px originales) sean coherentes.

Se agrupan los IoU por bucket y se promedia → mIoU_small / mIoU_medium / mIoU_large.
La operación por instancia está vectorizada (numpy booleano sobre el recorte del bbox);
el bucle externo sobre instancias es inevitable porque cada IoU es por instancia.

Uso (desde la raíz del repo):
    python tools/miou_by_size.py --ckpt checkpoints/train50/best.pt --coco-root /home/datasets/coco
    python tools/miou_by_size.py --ckpt checkpoints/train50/best.pt --coco-root /home/datasets/coco --tta
    python tools/miou_by_size.py --ckpt ... --coco-root ... --limit 500   # prueba rápida
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# permitir imports del repo al ejecutar desde tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes import COCO_CAT_ID_TO_INDEX            # noqa: E402
from config import Config                            # noqa: E402
from evaluate import carregar_checkpoint, maybe_wrap_with_tta  # noqa: E402
from models.unet import UNet                         # noqa: E402
from transforms import PairedTransform               # noqa: E402

COCO_SMALL_MAX  = 32 ** 2   # 1024
COCO_MEDIUM_MAX = 96 ** 2   # 9216
_SPLIT_DIR = {"train": "train2017", "val": "val2017"}


def _bucket(area: float) -> str:
    if area < COCO_SMALL_MAX:
        return "small"
    if area < COCO_MEDIUM_MAX:
        return "medium"
    return "large"


@torch.no_grad()
def miou_by_size(ckpt_path, coco_root, split="val", device=None,
                 use_tta=False, limit=None):
    """Devuelve {'small': (mIoU, n), 'medium': (mIoU, n), 'large': (mIoU, n)}."""
    from pycocotools.coco import COCO

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # ── modelo desde el checkpoint (backbone/num_classes del propio ckpt) ──
    state_dict, saved_cfg = carregar_checkpoint(ckpt_path)
    backbone    = saved_cfg.get("BACKBONE",    cfg.BACKBONE)
    num_classes = saved_cfg.get("NUM_CLASSES", cfg.NUM_CLASSES)
    img_size    = saved_cfg.get("IMG_SIZE",    getattr(cfg, "IMG_SIZE", 256))
    model = UNet(num_classes=num_classes, backbone=backbone, pretrained=False,
                 decoder_dropout=getattr(cfg, "DECODER_DROPOUT", 0.0)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    if use_tta:
        cfg.USE_TTA = True
    model = maybe_wrap_with_tta(model, cfg)
    print(f"[miou_by_size] device={device} backbone={backbone} num_classes={num_classes} "
          f"img_size={img_size} tta={use_tta}")

    # ── anotaciones COCO ──
    split_dir = _SPLIT_DIR[split]
    ann_file  = os.path.join(coco_root, "annotations", f"instances_{split_dir}.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"No existe el fichero de anotaciones: {ann_file}")
    coco    = COCO(ann_file)
    img_dir = Path(coco_root) / split_dir

    tf = PairedTransform(img_size=img_size, train=False, cfg=cfg)
    ious = {"small": [], "medium": [], "large": []}

    img_ids = list(coco.imgs.keys())
    if limit:
        img_ids = img_ids[:limit]

    for img_id in tqdm(img_ids, desc=f"miou_by_size {split}"):
        info = coco.loadImgs(img_id)[0]
        H, W = info["height"], info["width"]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = coco.loadAnns(ann_ids)
        if not anns:
            continue

        # inferencia → pred semántica (256) → reescalado a (H, W) con nearest
        img  = Image.open(img_dir / info["file_name"]).convert("RGB")
        x, _ = tf(img, Image.new("L", img.size, 0))
        pred_small = model(x.unsqueeze(0).to(device))[0].argmax(dim=0)         # (img_size, img_size)
        pred = F.interpolate(pred_small[None, None].float(), size=(H, W),
                             mode="nearest")[0, 0].cpu().numpy().astype(np.int32)

        for ann in anns:
            cls = COCO_CAT_ID_TO_INDEX.get(ann["category_id"])
            if cls is None:
                continue
            m = coco.annToMask(ann).astype(bool)        # (H, W) resolución original
            if not m.any():
                continue

            # recorte al bbox para localizar la predicción de la clase a esta instancia
            x0f, y0f, wf, hf = ann["bbox"]
            x0 = max(int(np.floor(x0f)), 0); y0 = max(int(np.floor(y0f)), 0)
            x1 = min(int(np.ceil(x0f + wf)), W); y1 = min(int(np.ceil(y0f + hf)), H)
            if x1 <= x0 or y1 <= y0:
                continue

            sub_gt   = m[y0:y1, x0:x1]                  # GT de la instancia (dentro del bbox)
            sub_pred = (pred[y0:y1, x0:x1] == cls)       # predicción de la clase c en el bbox
            inter = np.logical_and(sub_gt, sub_pred).sum()
            union = sub_gt.sum() + sub_pred.sum() - inter
            iou   = float(inter) / float(union) if union > 0 else 0.0

            ious[_bucket(float(ann["area"]))].append(iou)

    results = {}
    for k in ("small", "medium", "large"):
        v = ious[k]
        results[k] = (float(np.mean(v)) if v else float("nan"), len(v))
    return results


def main():
    p = argparse.ArgumentParser(description="mIoU estratificado por tamaño (COCO small/medium/large)")
    p.add_argument("--ckpt",      required=True, help="Checkpoint a evaluar")
    p.add_argument("--coco-root", required=True, help="Raíz de COCO (con annotations/ e imágenes)")
    p.add_argument("--split",     choices=["train", "val"], default="val")
    p.add_argument("--device",    default=None, help="cuda | cpu (auto si no se pasa)")
    p.add_argument("--tta",       action="store_true", help="Activa TTA (cfg.TTA_SCALES + hflip)")
    p.add_argument("--limit",     type=int, default=None,
                   help="Procesa solo las primeras N imágenes (prueba rápida)")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else None
    res = miou_by_size(args.ckpt, args.coco_root, split=args.split,
                       device=device, use_tta=args.tta, limit=args.limit)

    n_total = sum(n for _, n in res.values())
    print(f"\n=== mIoU por tamaño de objeto — COCO {args.split} "
          f"({n_total} instancias){'  [TTA]' if args.tta else ''} ===")
    print(f"{'bucket':<8} {'rango área (px)':<18} {'mIoU':>8} {'#inst':>8}")
    print("-" * 46)
    rangos = {"small": "< 1024", "medium": "1024–9216", "large": "≥ 9216"}
    for k in ("small", "medium", "large"):
        miou, n = res[k]
        miou_s = f"{miou:.4f}" if not np.isnan(miou) else "  n/a "
        print(f"{k:<8} {rangos[k]:<18} {miou_s:>8} {n:>8}")
    print("-" * 46)


if __name__ == "__main__":
    main()
