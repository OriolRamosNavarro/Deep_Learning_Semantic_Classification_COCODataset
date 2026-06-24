"""Avaluació final + visualització qualitativa (VOC2012 / COCO).

Carrega un checkpoint, calcula el mIoU sobre el split de validació i guarda una
figura amb N exemples:

    imatge original | ground truth | predicció del model | encerts / errors

El **backbone** i el **nombre de classes** es llegeixen del propi checkpoint (els
guarda main.py dins de `ckpt["config"]`), així funciona encara que `config.py`
hagi canviat des de l'entrenament (p.ex. tornar a posar resnet50 després d'haver
entrenat amb resnet152). Si el checkpoint és antic i no els porta, s'usa config.py.

Per a COCO, els exemples de la figura NO són les N primeres imatges (moltes de COCO
són gairebé tot fons i donarien una figura pobra): es mostregen unes quantes imatges
del split de validació, es puntua cada una per quant primer pla / quantes classes té,
i es trien les millors. És determinista (--seed).

Ús:
    # mètriques + figura (8 exemples) sobre COCO val
    python evaluate.py --ckpt checkpoints/best.pt --data-root /home/datasets/coco --num-samples 8

    # només la figura, amb nom propi
    python evaluate.py --ckpt checkpoints/best.pt --data-root /home/datasets/coco \
        --no-metrics --num-samples 6 --out docs/qual_r152_frozen.png

    # només mètriques (sense generar imatge)
    python evaluate.py --ckpt checkpoints/best.pt --data-root /home/datasets/coco --no-figure
"""
import argparse
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader

from classes import get_classes, get_colormap
from config import Config
from engine import validar
from losses import SegmentationLoss
from main import construir_dataset
from metrics import SegmentationMetrics
from models.unet import UNet
from transforms import PairedTransform


# ───────────────────────────────────────── TTA wrapper ─────────────────────────────────────
class TTAWrapper(torch.nn.Module):
    """
    EXPLICACIÓ SIMPLE: Test-Time Augmentation. Embolcalla un model i, en
    inferència, fa varies passades amb augmentations geomètriques (flips +
    multi-escala), promitja els logits, i retorna el resultat. No modifica
    el model original; només es fa servir en eval. Cost ≈ N passades extra
    on N = (1 + use_hflip) * len(scales).
    """
    def __init__(self, model: torch.nn.Module, scales=(1.0,), use_hflip: bool = False):
        super().__init__()
        self.model     = model
        self.scales    = tuple(scales) if scales else (1.0,)
        self.use_hflip = bool(use_hflip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Devuelve logits promedio en el tamaño de x.
        original_hw = x.shape[-2:]
        accumulated, n_views = None, 0

        for scale in self.scales:
            x_scaled = (x if scale == 1.0
                        else F.interpolate(x, scale_factor=scale,
                                           mode="bilinear", align_corners=False))

            logits = self.model(x_scaled)
            if logits.shape[-2:] != original_hw:
                logits = F.interpolate(logits, size=original_hw,
                                       mode="bilinear", align_corners=False)
            accumulated = logits if accumulated is None else accumulated + logits
            n_views += 1

            if self.use_hflip:
                logits_f = self.model(torch.flip(x_scaled, dims=[-1]))
                if logits_f.shape[-2:] != original_hw:
                    logits_f = F.interpolate(logits_f, size=original_hw,
                                             mode="bilinear", align_corners=False)
                accumulated = accumulated + torch.flip(logits_f, dims=[-1])
                n_views += 1

        return accumulated / max(1, n_views)


def maybe_wrap_with_tta(model, cfg):
    """Si cfg.USE_TTA está activo envuelve el model con TTAWrapper; si no, lo devuelve tal cual."""
    if not getattr(cfg, "USE_TTA", False):
        return model
    scales    = getattr(cfg, "TTA_SCALES", (1.0,))
    use_hflip = getattr(cfg, "TTA_HFLIP", True)
    n_views = (2 if use_hflip else 1) * len(scales)
    print(f"[evaluate] TTA activado: scales={scales} hflip={use_hflip} → {n_views} pasadas/imagen")
    return TTAWrapper(model, scales=scales, use_hflip=use_hflip)


# ───────────────────────────────────────── helpers visuals ─────────────────────────────────
def coloritzar_mascara(mask: np.ndarray, colormap) -> np.ndarray:
    """
    EXPLICACIÓ SIMPLE: Converteix una màscara d'índexs de classe (números) a una imatge RGB.
    Cada classe rep un color del colormap; els píxels ignorats (255, només a VOC) es pinten blancs.
    """
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(colormap):
        rgb[mask == cls_idx] = color
    rgb[mask == 255] = (255, 255, 255)
    return rgb


def mapa_encerts(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    EXPLICACIÓ SIMPLE: Imatge que ressalta on el model encerta i on falla respecte el ground truth.
    Verd = píxel correcte, vermell = píxel equivocat, blanc = píxel ignorat (255, VOC).
    En COCO la majoria de píxels són fons; si surten gairebé tot verd, és bon senyal.
    """
    valid = gt != 255
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[valid & (pred == gt)] = (0, 170, 0)
    out[valid & (pred != gt)] = (220, 0, 0)
    out[~valid] = (255, 255, 255)
    return out


def denormalitzar(image_t: torch.Tensor) -> np.ndarray:
    """
    EXPLICACIÓ SIMPLE: Desfà la normalització ImageNet (mitjana/desviació) per poder veure la imatge.
    """
    mean = torch.tensor(PairedTransform.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(PairedTransform.IMAGENET_STD).view(3, 1, 1)
    img  = image_t.detach().cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _to_np_mask(mask) -> np.ndarray:
    """La màscara pot venir com a Tensor (cas normal) o PIL/ndarray."""
    return mask.numpy() if torch.is_tensor(mask) else np.asarray(mask)


def triar_exemples(dataset, num_samples: int, scan_limit: int = 300, seed: int = 0):
    """
    EXPLICACIÓ SIMPLE: Tria quins índexs del dataset surten a la figura.
    Mostreja fins a `scan_limit` imatges, puntua cada una per (nº de classes diferents en
    primer pla) + (fracció de píxels en primer pla, topada a 0.5) i retorna els `num_samples`
    millors. Així evitem que la figura de COCO surti plena d'imatges quasi tot fons.
    Determinista: amb el mateix `seed` sempre tria els mateixos.
    """
    n = len(dataset)
    if n <= num_samples:
        return list(range(n))
    rng  = random.Random(seed)
    cand = rng.sample(range(n), min(scan_limit, n))
    puntuats = []
    for i in cand:
        _, mask = dataset[i]
        m  = _to_np_mask(mask)
        fg = m[(m != 0) & (m != 255)]
        if fg.size == 0:
            continue
        score = len(np.unique(fg)) + min(fg.size / m.size, 0.5)
        puntuats.append((score, i))
    if not puntuats:                       # cap candidat amb primer pla → els primers
        return list(range(num_samples))
    puntuats.sort(reverse=True)
    return sorted(i for _, i in puntuats[:num_samples])


@torch.no_grad()
def fer_figura(model, dataset, device, idxs, classes, colormap, out_path):
    """
    EXPLICACIÓ SIMPLE: Crea i guarda la figura amb una fila per exemple i 4 columnes:
    input | ground truth | predicció | encerts/errors. A sota, una llegenda amb les classes
    de primer pla que apareixen als exemples (índex + nom + color), perquè s'entenguin els colors.
    """
    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]

    model.eval()
    classes_presents = set()
    for row, idx in enumerate(idxs):
        image, mask = dataset[idx]
        gt   = _to_np_mask(mask)
        pred = model(image.unsqueeze(0).to(device))[0].argmax(dim=0).cpu().numpy()

        classes_presents.update(int(c) for c in np.unique(gt[gt != 255]) if c != 0)
        classes_presents.update(int(c) for c in np.unique(pred) if c != 0)

        cols = (
            ("input",            denormalitzar(image)),
            ("ground truth",     coloritzar_mascara(gt, colormap)),
            ("prediction",       coloritzar_mascara(pred, colormap)),
            ("encerts / errors", mapa_encerts(gt, pred)),
        )
        for c, (title, im) in enumerate(cols):
            axes[row, c].imshow(im)
            axes[row, c].set_title(title if row == 0 else "")
            axes[row, c].axis("off")

    handles = [
        Patch(facecolor=np.array(colormap[c]) / 255.0, edgecolor="k",
              label=f"{c}: {classes[c] if c < len(classes) else c}")
        for c in sorted(classes_presents)
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", frameon=False,
                   ncol=min(6, len(handles)), fontsize=8, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] figura guardada a {out_path}  ({n} exemples; índexs {idxs})")


@torch.no_grad()
def inferir_fotos_custom(model, device, cfg, classes, colormap, in_dir, out_dir):
    """
    EXPLICACIÓ SIMPLE: Corre el model sobre fotos PRÒPIES (sense ground truth) d'una
    carpeta i en guarda, per cada foto, una figura: original | predicció | overlay.
    Com que no hi ha màscara real, NO es calcula mIoU; només inferència + visualització.
    A stdout imprimeix les classes detectades a cada foto amb el seu % de píxels.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de fotos custom: {in_dir}")
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"No hay imágenes ({'/'.join(sorted(exts))}) en {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = getattr(cfg, "IMG_SIZE", 256)
    tf = PairedTransform(img_size=img_size, train=False, cfg=cfg)
    model.eval()
    print(f"\n=== fotos custom: {len(paths)} imágenes en {in_dir} ===")

    for p in paths:
        img   = Image.open(p).convert("RGB")
        dummy = Image.new("L", img.size, 0)               # máscara ficticia (no hay GT)
        x, _  = tf(img, dummy)
        pred  = model(x.unsqueeze(0).to(device))[0].argmax(dim=0).cpu().numpy()

        input_rgb = denormalitzar(x)
        pred_rgb  = coloritzar_mascara(pred, colormap)
        overlay   = (0.55 * input_rgb + 0.45 * pred_rgb).clip(0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        for ax, title, im in zip(axes,
                                 ("input", "predicción", "overlay"),
                                 (input_rgb, pred_rgb, overlay)):
            ax.imshow(im); ax.set_title(title); ax.axis("off")

        # leyenda + resumen de clases detectadas (por % de píxeles, sin fondo)
        ids, counts = np.unique(pred, return_counts=True)
        total = pred.size
        detected = sorted(((int(c), n) for c, n in zip(ids, counts) if c != 0),
                          key=lambda t: t[1], reverse=True)
        handles = [Patch(facecolor=np.array(colormap[c]) / 255.0, edgecolor="k",
                         label=f"{classes[c] if c < len(classes) else c}  {100*n/total:.0f}%")
                   for c, n in detected]
        if handles:
            fig.legend(handles=handles, loc="lower center", frameon=False,
                       ncol=min(6, len(handles)), fontsize=8, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(p.name)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
        out_path = out_dir / f"{p.stem}_seg.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        resumen = ", ".join(f"{classes[c] if c < len(classes) else c} {100*n/total:.0f}%"
                            for c, n in detected[:5]) or "(solo fondo)"
        print(f"  {p.name:<32} → {out_path.name}   [{resumen}]")

    print(f"[evaluate] {len(paths)} figuras guardadas en {out_dir}")


# ───────────────────────────────────────── checkpoint ──────────────────────────────────────
def carregar_checkpoint(ckpt_path: str):
    """
    EXPLICACIÓ SIMPLE: Llegeix el fitxer del checkpoint i en treu:
    - els pesos del model (state_dict), netejant el prefix '_orig_mod.' que afegeix torch.compile
    - la config que es va guardar a l'entrenament (backbone, nº classes, dataset...) si hi és
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        saved_cfg  = ckpt.get("config", {}) or {}
        if "mIoU" in ckpt:
            print(f"[evaluate] checkpoint @ epoch {ckpt.get('epoch', '?')}  "
                  f"(mIoU guardat = {ckpt['mIoU']:.4f})")
    else:                                  # checkpoint "pelat": només el state_dict
        state_dict, saved_cfg = ckpt, {}
    state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                  for k, v in state_dict.items()}
    return state_dict, saved_cfg


# ───────────────────────────────────────── principal ───────────────────────────────────────
def principal(args):
    """
    EXPLICACIÓ SIMPLE: Avalua un model entrenat.
    1. Llegeix el checkpoint (i d'allà backbone + nº classes + dataset).
    2. Reconstrueix la U-Net i hi carrega els pesos.
    3. (si no --no-metrics) calcula mIoU i IoU per classe sobre el split de validació.
    4. (si no --no-figure) genera la figura qualitativa input | GT | pred | encerts.
    """
    cfg = Config()
    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, saved_cfg = carregar_checkpoint(args.ckpt)
    backbone     = saved_cfg.get("BACKBONE",    cfg.BACKBONE)
    num_classes  = saved_cfg.get("NUM_CLASSES", cfg.NUM_CLASSES)
    dataset_name = saved_cfg.get("DATASET",     cfg.DATASET)
    print(f"[evaluate] device = {device}  |  dataset = {dataset_name}  |  "
          f"backbone = {backbone}  |  num_classes = {num_classes}")
    if (backbone, num_classes) != (cfg.BACKBONE, cfg.NUM_CLASSES):
        print(f"[evaluate] (config.py actual diu backbone={cfg.BACKBONE}, "
              f"num_classes={cfg.NUM_CLASSES}; faig servir els valors del checkpoint)")

    # En modo --custom no hace falta el split de validación (no hay GT que cargar).
    val_ds = None if args.custom else construir_dataset(cfg, args.data_root, "val")

    # build_model construye UNet o DeepLabV3+ según cfg.DECODER_TYPE.
    from models import build_model
    cfg.BACKBONE   = backbone
    cfg.PRETRAINED = False     # los pesos vienen del checkpoint, no de ImageNet
    model = build_model(num_classes=num_classes, cfg=cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Si --tta o cfg.USE_TTA, envolvemos el modelo con TTAWrapper.
    # El flag CLI --tta sobreescribe el cfg.
    if getattr(args, "tta", False):
        cfg.USE_TTA = True
    eval_model = maybe_wrap_with_tta(model, cfg)

    classes  = get_classes(dataset_name)
    colormap = get_colormap(dataset_name)

    # ── modo fotos propias: inferencia sobre una carpeta y salida; no toca val ──
    if args.custom:
        inferir_fotos_custom(eval_model, device, cfg, classes, colormap,
                             args.custom_dir, args.custom_out)
        return

    if not args.no_metrics:
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                num_workers=cfg.NUM_WORKERS, pin_memory=True)
        # La loss de val se construye con la config guardada en el checkpoint para que
        # `val_loss` coincida con la del entrenamiento; si el checkpoint no la trae,
        # se cae al config.py actual.
        def _from_saved(key, default):
            return saved_cfg.get(key, getattr(cfg, key, default))
        loss_weights = {
            "ce":          _from_saved("CE_WEIGHT",          0.0),
            "dice":        _from_saved("DICE_WEIGHT",        0.5),
            "focal":       _from_saved("FOCAL_WEIGHT",       0.5),
            "lovasz":      _from_saved("LOVASZ_WEIGHT",      0.0),
            "ohem_ce":     _from_saved("OHEM_CE_WEIGHT",     0.0),
            "weighted_ce": _from_saved("WEIGHTED_CE_WEIGHT", 0.0),
        }
        # Si CLASS_WEIGHTS == "auto" y no hay cache en disco, caemos a None para no
        # crashear evaluate.py (la val_loss no será 100% comparable, pero el mIoU sí).
        cw = _from_saved("CLASS_WEIGHTS", None)
        if isinstance(cw, str) and cw.lower() == "auto":
            cache = os.path.join("checkpoints", f"class_weights_auto_{num_classes}cls.pt")
            if not os.path.isfile(cache):
                print(f"[evaluate] CLASS_WEIGHTS='auto' sin cache en {cache} → se usa None "
                      "para no abortar la evaluación.")
                cw = None
        criterion = SegmentationLoss(
            weights       = loss_weights,
            ignore_index  = _from_saved("IGNORE_INDEX", 255),
            num_classes   = num_classes,
            focal_gamma   = _from_saved("FOCAL_GAMMA", 2.0),
            ohem_top_k    = _from_saved("OHEM_TOP_K", 0.25),
            class_weights = cw,
            label_smoothing=_from_saved("LABEL_SMOOTHING", 0.0),
        )
        metrics = SegmentationMetrics(
            num_classes=num_classes, ignore_index=cfg.IGNORE_INDEX,
            compute_pixel_accuracy=getattr(cfg, "LOG_PIXEL_ACCURACY", False),
            compute_f1            =getattr(cfg, "LOG_F1_PER_CLASS",   False),
            compute_boundary_iou  =getattr(cfg, "LOG_BOUNDARY_IOU",   False),
        )
        val_loss, val_metrics = validar(eval_model, val_loader, criterion, metrics, device)

        if args.dump_cm:
            cm = metrics.confusion_matrix.detach().cpu().clone()
            torch.save({"confusion_matrix": cm,
                        "class_names": classes,
                        "mIoU": float(val_metrics["mIoU"])},
                       args.dump_cm)
            print(f"[evaluate] confusion matrix volcada → {args.dump_cm}")

        iou = val_metrics["IoU_per_class"]
        print(f"\n=== {dataset_name} val ===")
        print(f"val_loss : {val_loss:.4f}")
        print(f"mIoU     : {val_metrics['mIoU']:.4f}   (mitjana sobre les classes presents al GT)")
        print(f"\nIoU per classe (ascendent; 0.0000 ≈ classe absent o mai predita):")
        for name, v in sorted(zip(classes, iou), key=lambda x: x[1]):
            print(f"  {name:<22} {v:.4f}")

    if not args.no_figure:
        idxs = triar_exemples(val_ds, args.num_samples, scan_limit=args.scan_limit, seed=args.seed)
        fer_figura(eval_model, val_ds, device, idxs, classes, colormap, args.out)


def analitzar_arguments():
    """
    EXPLICACIÓ SIMPLE: Arguments de línia de comandes per a evaluate.py.
    """
    p = argparse.ArgumentParser(description="Avalua un checkpoint U-Net (VOC2012 / COCO)")
    p.add_argument("--ckpt",        type=str, default="checkpoints/best.pt",
                   help="Ruta del checkpoint a avaluar")
    p.add_argument("--data-root",   type=str, default="./data",
                   help="Arrel del dataset (carpeta arrel de COCO si DATASET='COCO')")
    p.add_argument("--num-samples", type=int, default=8, help="Nº d'exemples a la figura")
    p.add_argument("--out",         type=str, default="docs/qualitative_results.png",
                   help="Ruta de sortida de la figura")
    p.add_argument("--scan-limit",  type=int, default=300,
                   help="Quantes imatges de val es mostregen per triar les més variades "
                        "(baixa-ho si val no té màscares pre-generades i va lent)")
    p.add_argument("--seed",        type=int, default=0, help="Llavor per triar els exemples")
    p.add_argument("--device",      type=str, default=None, help="cuda | cpu (auto si no es passa)")
    p.add_argument("--no-figure",   action="store_true", help="No generar la figura (només mètriques)")
    p.add_argument("--no-metrics",  action="store_true", help="No calcular mIoU (només la figura)")
    p.add_argument("--tta",         action="store_true",
                   help="Activa TTA (multi-escala + hflip) usando cfg.TTA_SCALES y cfg.TTA_HFLIP")
    p.add_argument("--dump-cm",     type=str, default=None,
                   help="Vuelca la confusion matrix de validación a este .pt "
                        "(formato leído por make_plots.py confusion_matrix)")
    p.add_argument("--custom",      action="store_true",
                   help="Modo fotos propias: corre inferencia sobre las imágenes de "
                        "--custom-dir (sin GT, sin mIoU) y guarda input|predicción|overlay")
    p.add_argument("--custom-dir",  type=str, default="fotos_custom",
                   help="Carpeta con las fotos propias (modo --custom)")
    p.add_argument("--custom-out",  type=str, default="docs/custom_preds",
                   help="Carpeta de salida de las figuras (modo --custom)")
    return p.parse_args()


if __name__ == "__main__":
    principal(analitzar_arguments())
