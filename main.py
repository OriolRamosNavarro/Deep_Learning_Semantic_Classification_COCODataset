import os
os.environ.setdefault("TORCH_HOME", r"C:\torch_cache")

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from classes import get_classes
from coco_filter import masks_dirname
from config import Config
from dataset import CocoSegmentation, CocoSegmentationCached
from ema import EMA
from engine import entrenar_una_epoca, validar
from losses import SegmentationLoss
from metrics import SegmentationMetrics
from models.unet import UNet
from models import build_model
from transforms import PairedTransform


def establir_llavor(seed: int) -> None:
    """Fija las semillas aleatorias para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def construir_dataset(cfg: Config, root: str, split: str):
    """Construye el dataset (train/val) según cfg.DATASET.

    - VOC : torchvision.datasets.VOCSegmentation (se descarga solo).
    - COCO: CocoSegmentationCached si hay máscaras pre-generadas
            (ver tools/precompute_coco_masks.py); si no, CocoSegmentation
            (genera las máscaras al vuelo — mucho más lento).
    """
    transform = PairedTransform(img_size=cfg.IMG_SIZE, train=(split == "train"), cfg=cfg)
    dataset = cfg.DATASET.upper()

    if dataset in ("VOC", "VOC2012"):
        image_set = "train" if split == "train" else "val"
        return datasets.VOCSegmentation(root=root, year="2012", image_set=image_set,
                                        download=True, transforms=transform)

    if dataset == "COCO":
        instance_sizes = getattr(cfg, "INSTANCE_SIZES", None)
        masks_root = getattr(cfg, "MASKS_ROOT", None) or root
        split_dir  = "train2017" if split == "train" else "val2017"
        # carpeta de máscaras según el filtro de tamaño (sufijo si se filtra)
        cached_dir = os.path.join(masks_root, masks_dirname(split_dir, instance_sizes))
        if os.path.isdir(cached_dir):
            return CocoSegmentationCached(root=root, split=split, transforms=transform,
                                          masks_root=masks_root, instance_sizes=instance_sizes)
        print(f"[main] Aviso: no hay máscaras pre-generadas en {cached_dir}; se usa "
              f"CocoSegmentation (lento). Genera las máscaras con tools/precompute_coco_masks.py "
              f"(usa --masks-root si el COCO es de solo lectura, y --instance-sizes si filtras).")
        return CocoSegmentation(root=root, split=split, transforms=transform,
                                instance_sizes=instance_sizes)

    raise ValueError(f"DATASET desconocido: {cfg.DATASET!r}. Usa 'VOC' o 'COCO'.")


def construir_optimitzador(model: UNet, cfg: Config) -> torch.optim.Optimizer:
    """Crea el optimizer con learning rates separados para encoder (bajo) y decoder (alto)."""
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith("encoder.") and p.requires_grad]
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": cfg.LR_ENCODER})
    param_groups.append({"params": decoder_params, "lr": cfg.LR_DECODER})

    name = cfg.OPTIMIZER.lower()
    if name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    if name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=cfg.SGD_MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    if name == "rmsprop":
        return torch.optim.RMSprop(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    if name == "adagrad":
        return torch.optim.Adagrad(param_groups, weight_decay=cfg.WEIGHT_DECAY)
    raise ValueError(f"Optimizer desconocido: {cfg.OPTIMIZER!r}. "
                     f"Usa: adamw | adam | sgd | rmsprop | adagrad")


def build_scheduler(optimizer, cfg, warmup_steps, total_steps,
                    steps_per_epoch=None):
    """Construye el scheduler según cfg.SCHEDULER. Todos opera per-batch step.

      - "cosine_warmup" (default): warmup lineal + cosine annealing.
      - "poly": warmup lineal + decaimiento polinomial (1 - t)^POLY_POWER.
      - "step": warmup lineal + decay multiplicativo cada STEP_SIZE epochs.
      - "constant": warmup lineal + LR constante (sin decay).
    """
    sched = getattr(cfg, "SCHEDULER", "cosine_warmup").lower()

    if sched == "cosine_warmup":
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if sched == "poly":
        power = getattr(cfg, "POLY_POWER", 0.9)
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, (1.0 - progress) ** power)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if sched == "step":
        if steps_per_epoch is None or steps_per_epoch <= 0:
            raise ValueError("scheduler 'step' requiere steps_per_epoch > 0")
        step_epochs = getattr(cfg, "STEP_SIZE", 30)
        gamma       = getattr(cfg, "STEP_GAMMA", 0.1)
        step_steps  = step_epochs * steps_per_epoch
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            decays = (step - warmup_steps) // max(1, step_steps)
            return gamma ** decays
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if sched == "constant":
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"SCHEDULER desconocido: {cfg.SCHEDULER!r}. "
                     "Usa: cosine_warmup | poly | step | constant")


def registre_iou_per_classe(iou_per_classe, class_names, prefix="val_iou"):
    """Dict {val_iou/<clase>: iou} para Wandb (omite las clases 'N/A' de COCO)."""
    return {f"{prefix}/{name}": float(iou)
            for name, iou in zip(class_names, iou_per_classe)
            if name != "N/A"}


def _parse_class_weights_flag(s):
    """Convierte el string del flag --class-weights a lo que espera SegmentationLoss.
       'none' → None, 'auto' → 'auto', 'a,b,c,...' → [float, ...]."""
    s = s.strip()
    if s.lower() == "none":
        return None
    if s.lower() == "auto":
        return "auto"
    return [float(x) for x in s.split(",")]


def _build_loss_str(weights, focal_gamma, ohem_top_k, class_weights):
    """Identificador compacto de la loss para el run_name de Wandb.

    Formato: <nombre><peso> separado por '_' SOLO con las losses activas
    (peso > 0). Anexa hiperparámetros solo si no son default:
        gamma != 2.0          → "_g<gamma>"
        ohem_top_k != 0.25    → "_k<top_k>"
        class_weights != None → "_cw"
    Ejemplos:
        {"ce":1.0}                            → "ce1"
        {"focal":0.5,"dice":0.5}              → "focal0.5_dice0.5"
        {"focal":0.5,"lovasz":0.5}, γ=3       → "focal0.5_lovasz0.5_g3"
        {"ohem_ce":1.0}, k=0.2                → "ohem_ce1_k0.2"
        {"weighted_ce":1.0}, cw='auto'        → "weighted_ce1_cw"
    """
    order = ("ce", "dice", "focal", "lovasz", "ohem_ce", "weighted_ce")
    parts = [f"{n}{weights[n]:g}" for n in order if weights.get(n, 0.0) > 0]
    suffix = ""
    if weights.get("focal", 0.0) > 0 and focal_gamma != 2.0:
        suffix += f"_g{focal_gamma:g}"
    if weights.get("ohem_ce", 0.0) > 0 and ohem_top_k != 0.25:
        suffix += f"_k{ohem_top_k:g}"
    if weights.get("weighted_ce", 0.0) > 0 and class_weights is not None:
        suffix += "_cw"
    return "_".join(parts) + suffix


def _apply_loss_overrides(cfg, args):
    """Sobrescribe los atributos de cfg con los flags CLI si vienen != None.
       Ha de pasarse ANTES de instanciar la loss para que cfg/W&B reflejen los
       valores efectivos. Tolera namespaces sin todos los flags (p.ej. fast_train.py)."""
    overrides = (
        ("ce_weight",          "CE_WEIGHT"),
        ("dice_weight",        "DICE_WEIGHT"),
        ("focal_weight",       "FOCAL_WEIGHT"),
        ("lovasz_weight",      "LOVASZ_WEIGHT"),
        ("ohem_ce_weight",     "OHEM_CE_WEIGHT"),
        ("weighted_ce_weight", "WEIGHTED_CE_WEIGHT"),
        ("focal_gamma",        "FOCAL_GAMMA"),
        ("ohem_top_k",         "OHEM_TOP_K"),
    )
    for flag, attr in overrides:
        val = getattr(args, flag, None)
        if val is not None:
            setattr(cfg, attr, val)
    cw_flag = getattr(args, "class_weights", None)
    if cw_flag is not None:
        cfg.CLASS_WEIGHTS = _parse_class_weights_flag(cw_flag)


def principal(args: argparse.Namespace) -> None:
    cfg = Config()
    _apply_loss_overrides(cfg, args)        # los flags CLI tienen prioridad sobre cfg
    establir_llavor(cfg.SEED)
    if getattr(cfg, "CUDNN_BENCHMARK", False):
        torch.backends.cudnn.benchmark = True   # tamaño de input fijo → cuDNN puede autotunear
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device = {device}  |  dataset = {cfg.DATASET}  |  num_classes = {cfg.NUM_CLASSES}")

    # ── datos ────────────────────────────────────────────────────────────────
    train_ds = construir_dataset(cfg, args.data_root, "train")
    val_ds   = construir_dataset(cfg, args.data_root, "val")

    if args.overfit > 0:
        idx = list(range(args.overfit))
        train_ds = Subset(train_ds, idx)
        val_ds   = train_ds   # mismas imágenes en val: queremos ver overfit
        print(f"[main] OVERFIT mode on {args.overfit} samples")

    loader_kwargs = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True)
    if cfg.NUM_WORKERS > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=getattr(cfg, "PREFETCH_FACTOR", 2))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, **loader_kwargs)

    # ── modelo ───────────────────────────────────────────────────────────────
    # build_model elige UNet o DeepLabV3+ según cfg.DECODER_TYPE.
    model = build_model(num_classes=cfg.NUM_CLASSES, cfg=cfg).to(device)
    arch  = str(getattr(cfg, "DECODER_TYPE", "unet")).lower()
    print(f"[main] arquitectura: {arch}  |  backbone: {cfg.BACKBONE}")

    freeze_map = {
        "layer0": cfg.FREEZE_LAYER0, "layer1": cfg.FREEZE_LAYER1, "layer2": cfg.FREEZE_LAYER2,
        "layer3": cfg.FREEZE_LAYER3, "layer4": cfg.FREEZE_LAYER4,
    }
    frozen = []
    for layer_name, should_freeze in freeze_map.items():
        if should_freeze:
            for param in getattr(model.encoder, layer_name).parameters():
                param.requires_grad = False
            frozen.append(layer_name)
    if frozen:
        print(f"[main] Capas congeladas: {', '.join(frozen)}")

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[main] {arch} params: {n_params/1e6:.2f}M total | {n_trainable/1e6:.2f}M entrenables")

    # ── optimizaciones de velocidad (solo en GPU) ────────────────────────────
    use_amp        = getattr(cfg, "USE_AMP", False) and device.type == "cuda"
    channels_last  = getattr(cfg, "CHANNELS_LAST", False) and device.type == "cuda"
    grad_clip_norm = getattr(cfg, "GRAD_CLIP_NORM", 0.0)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("[main] channels_last activado")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("[main] AMP (mixed precision fp16) activado")

    # torch.compile acelera forward/backward; mantenemos `model` sin compilar como
    # referencia para guardar el checkpoint (evita el prefijo '_orig_mod.' en el state_dict).
    # No se compila en modo --overfit: la compilación inicial no compensa con pocos pasos.
    train_model = model
    if getattr(cfg, "COMPILE", False) and device.type == "cuda" and args.overfit == 0:
        try:
            train_model = torch.compile(model)
            print("[main] torch.compile activado")
        except Exception as e:
            print(f"[main] torch.compile no disponible ({e}); se usa el modelo sin compilar")

    # ── EMA (Exponential Moving Average) de los pesos ────────────────────────
    use_ema = bool(getattr(cfg, "USE_EMA", False))
    ema = None
    if use_ema:
        ema = EMA(model, decay=getattr(cfg, "EMA_DECAY", 0.9999))
        print(f"[main] EMA activado (decay = {ema.decay})")

    # ── loss + optimizer + scheduler + métricas ──────────────────────────────
    loss_weights = {
        "ce":          cfg.CE_WEIGHT,
        "dice":        cfg.DICE_WEIGHT,
        "focal":       cfg.FOCAL_WEIGHT,
        "lovasz":      cfg.LOVASZ_WEIGHT,
        "ohem_ce":     cfg.OHEM_CE_WEIGHT,
        "weighted_ce": cfg.WEIGHTED_CE_WEIGHT,
    }
    needs_loader_for_auto = (cfg.WEIGHTED_CE_WEIGHT > 0 and cfg.CLASS_WEIGHTS == "auto")
    criterion = SegmentationLoss(
        weights         = loss_weights,
        ignore_index    = cfg.IGNORE_INDEX,
        num_classes     = cfg.NUM_CLASSES,
        focal_gamma     = cfg.FOCAL_GAMMA,
        ohem_top_k      = cfg.OHEM_TOP_K,
        class_weights   = cfg.CLASS_WEIGHTS,
        train_loader    = train_loader if needs_loader_for_auto else None,
        label_smoothing = getattr(cfg, "LABEL_SMOOTHING", 0.0),
    )
    print(f"[main] {criterion}")
    epochs    = args.epochs if args.epochs is not None else cfg.EPOCHS
    optimizer = construir_optimitzador(model, cfg)

    steps_per_epoch = max(1, len(train_loader))
    warmup_steps    = getattr(cfg, "WARMUP_EPOCHS", 0) * steps_per_epoch
    total_steps     = epochs * steps_per_epoch
    scheduler       = build_scheduler(optimizer, cfg, warmup_steps, total_steps,
                                      steps_per_epoch=steps_per_epoch)
    print(f"[main] scheduler = {getattr(cfg, 'SCHEDULER', 'cosine_warmup')}  "
          f"(warmup={getattr(cfg, 'WARMUP_EPOCHS', 0)} ep, total={epochs} ep)")

    metrics = SegmentationMetrics(
        num_classes=cfg.NUM_CLASSES,
        ignore_index=cfg.IGNORE_INDEX,
        compute_pixel_accuracy=getattr(cfg, "LOG_PIXEL_ACCURACY", False),
        compute_f1            =getattr(cfg, "LOG_F1_PER_CLASS",   False),
        compute_boundary_iou  =getattr(cfg, "LOG_BOUNDARY_IOU",   False),
    )
    # Métricas de train (solo mIoU, sin extras) → permite ver overfitting comparando
    # train_mIoU vs val_mIoU. Instancia aparte para no pisar la CM de validación
    # que se guarda en el checkpoint.
    train_metrics = SegmentationMetrics(
        num_classes=cfg.NUM_CLASSES,
        ignore_index=cfg.IGNORE_INDEX,
    )
    class_names = get_classes(cfg.DATASET)

    # ── wandb ────────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        frozen_layers = [f"L{i}" for i, f in enumerate(freeze_map.values()) if f]
        freeze_str = f"freeze({'_'.join(frozen_layers)})" if frozen_layers else "nofrozen"
        loss_str = _build_loss_str(loss_weights, cfg.FOCAL_GAMMA, cfg.OHEM_TOP_K, cfg.CLASS_WEIGHTS)
        # Run name: --wandb-run-name lo sobreescribe TODO; si no, se construye automático.
        auto_run_name = "overfit" if args.overfit > 0 else \
                        f"{cfg.DATASET.lower()}-{cfg.BACKBONE}-{cfg.OPTIMIZER}-{freeze_str}-{loss_str}"
        run_name = getattr(args, "wandb_run_name", None) or auto_run_name
        # Project: --wandb-project sobreescribe Config.WANDB_PROJECT.
        project = getattr(args, "wandb_project", None) or getattr(cfg, "WANDB_PROJECT", "finetuning")
        wandb.init(project=project, name=run_name,
                   mode="offline" if args.wandb_offline else "online",
                   config={k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
        wandb_log_what = getattr(cfg, "WANDB_LOG_GRADIENTS", "all")
        wandb.watch(model, criterion,
                    log=wandb_log_what if wandb_log_what else None,
                    log_freq=getattr(cfg, "WANDB_LOG_FREQ", 50))

    # ── bucle de entrenamiento ───────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir if args.ckpt_dir else getattr(cfg, "CKPT_DIR", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every_n = int(getattr(cfg, "SAVE_EVERY_N_EPOCHS", 0) or 0)
    best_miou = 0.0

    for epoch in range(epochs):
        train_loss = entrenar_una_epoca(train_model, train_loader, optimizer, criterion, device,
                                        scaler=scaler, scheduler=scheduler, use_amp=use_amp,
                                        channels_last=channels_last, grad_clip_norm=grad_clip_norm,
                                        epoch=epoch, ema=ema, metrics=train_metrics)
        train_mIoU = train_metrics.calcular()["mIoU"]

        # Si EMA está activo: validamos con los pesos suavizados y luego restauramos.
        if ema is not None:
            ema.apply_shadow(model)
        val_loss, val_metrics = validar(train_model, val_loader, criterion, metrics, device,
                                        use_amp=use_amp, channels_last=channels_last, epoch=epoch)
        if ema is not None:
            ema.restore(model)

        log = {
            "epoch":       epoch,
            "train_loss":  train_loss,
            "train_mIoU":  train_mIoU,
            "val_loss":    val_loss,
            "val_mIoU":    val_metrics["mIoU"],
            "lr_encoder":  optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
            "lr_decoder":  optimizer.param_groups[-1]["lr"],
        }
        log.update(registre_iou_per_classe(val_metrics["IoU_per_class"], class_names))
        # Métricas extra (solo si están activas en config).
        if "pixel_accuracy" in val_metrics:
            log["val_pixel_acc"] = val_metrics["pixel_accuracy"]
        if "mF1" in val_metrics:
            log["val_mF1"] = val_metrics["mF1"]
            log.update(registre_iou_per_classe(val_metrics["F1_per_class"], class_names,
                                               prefix="val_f1"))
        if "boundary_mIoU" in val_metrics:
            log["val_boundary_mIoU"] = val_metrics["boundary_mIoU"]

        main_keys = ("epoch", "train_loss", "train_mIoU", "val_loss", "val_mIoU")
        line = " | ".join(f"{k}={log[k]:.4f}" if isinstance(log[k], float) else f"{k}={log[k]}"
                          for k in main_keys)
        print(f"[epoch {epoch:03d}] {line}")

        if use_wandb:
            wandb.log(log)

        # Snapshot del estado a guardar (común al "best" y al "every N").
        # Si EMA está activo: el mIoU se ha medido con los pesos suavizados, así
        # que guardamos esos mismos pesos en el checkpoint (apply_shadow temporalmente
        # y luego restore). Si no, guardamos el state_dict normal.
        if ema is not None:
            ema.apply_shadow(model)
            ckpt_model_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.restore(model)
        else:
            ckpt_model_sd = model.state_dict()
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": ckpt_model_sd,        # pesos EMA si EMA activo, raw si no
            "mIoU": float(val_metrics["mIoU"]),
            "config": {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()},
            # CM + nombres de clase del epoch recién validado → make_plots.py los lee
            # directamente del checkpoint (sin necesidad de evaluate.py --dump-cm).
            "confusion_matrix": metrics.confusion_matrix.detach().cpu().clone(),
            "class_names": class_names,
        }

        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            ckpt_state["mIoU"] = best_miou
            torch.save(ckpt_state, ckpt_dir / "best.pt")
            print(f"[epoch {epoch:03d}] new best mIoU = {best_miou:.4f} → checkpoint guardado")

        if save_every_n > 0 and (epoch + 1) % save_every_n == 0:
            snap_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt_state, snap_path)
            print(f"[epoch {epoch:03d}] snapshot periódico → {snap_path}")

    print(f"[main] Done. Best mIoU = {best_miou:.4f}")
    if use_wandb:
        wandb.summary["best_mIoU"] = best_miou
        wandb.finish()


def analitzar_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="U-Net segmentation training (VOC2012 / COCO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", type=str, default="./data",
                   help="Carpeta donde descargar/leer el dataset (raíz de COCO si DATASET='COCO')")
    p.add_argument("--epochs", type=int, default=None, help="Sobrescribe Config.EPOCHS")
    p.add_argument("--overfit", type=int, default=0,
                   help="Si >0, entrena/valida sobre las primeras N imágenes (sanity check)")
    p.add_argument("--no-wandb", action="store_true", help="Desactiva Wandb")
    p.add_argument("--wandb-offline", action="store_true", help="Wandb en modo offline")
    p.add_argument("--wandb-project", type=str, default=None,
                   help="Nombre del proyecto en Wandb (override de Config.WANDB_PROJECT)")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="Nombre del run en Wandb (override del nombre autogenerado)")
    p.add_argument("--ckpt-dir", type=str, default=None,
                   help="Carpeta donde guardar checkpoints (override de Config.CKPT_DIR). "
                        "Útil cuando varias VM comparten $HOME por NFS y queremos evitar "
                        "que un sweep pise el best.pt de otro.")

    # ── pesos de la loss combinada (todos default None → usa el valor de cfg) ─
    g = p.add_argument_group("loss combinada (override de config.py si != None)")
    g.add_argument("--ce-weight",          type=float, default=None, help="peso CE")
    g.add_argument("--dice-weight",        type=float, default=None, help="peso Dice")
    g.add_argument("--focal-weight",       type=float, default=None, help="peso Focal")
    g.add_argument("--lovasz-weight",      type=float, default=None, help="peso Lovász-Softmax")
    g.add_argument("--ohem-ce-weight",     type=float, default=None, help="peso OHEM-CE")
    g.add_argument("--weighted-ce-weight", type=float, default=None, help="peso CE con pesos por clase")
    g.add_argument("--focal-gamma",        type=float, default=None, help="gamma de Focal (default cfg 2.0)")
    g.add_argument("--ohem-top-k",         type=float, default=None,
                   help="fracción de píxeles 'difíciles' para OHEM-CE (default cfg 0.25)")
    g.add_argument("--class-weights",      type=str,   default=None,
                   help="'none' | 'auto' | lista 'w0,w1,...' de NUM_CLASSES floats")
    return p.parse_args()


if __name__ == "__main__":
    principal(analitzar_arguments())
