"""
Pérdidas para segmentación semántica.

Hay 6 losses individuales y una clase `SegmentationLoss` que combina las que
se quieran como **suma ponderada**: cada loss recibe un peso ≥ 0 y solo se
calculan (y suman) las que tienen peso > 0. Si todos los pesos son 0 o el
dict está vacío → ValueError al construirla.

Losses soportadas
-----------------
- ``ce``           CrossEntropyLoss(ignore_index) "vainilla".
- ``dice``         Dice Loss (1 − Dice score), trabaja sobre **probs** (softmax).
- ``focal``        Focal Loss numéricamente estable a partir de logits;
                   ``gamma`` enfoca en píxeles difíciles.
- ``lovasz``       Lovász-Softmax (Berman et al. 2018), trabaja sobre probs;
                   surrogate diferenciable del IoU. Más lento que el resto.
- ``ohem_ce``      Online Hard Example Mining: CE solo sobre el top-k% de
                   píxeles con más loss (``top_k`` configurable).
- ``weighted_ce``  CE con pesos por clase (``class_weights``: ``None`` |
                   ``"auto"`` = frecuencia inversa cacheada del train set |
                   ``list[float]`` de ``num_classes``).

Convención de entrada/salida
----------------------------
- preds (a SegmentationLoss): **logits** ``(B, C, H, W)``.
- targets: ``(B, H, W)`` con índices de clase ``long`` (o ``ignore_index``).
- forward retorna un escalar (loss combinada).
- Internamente, SegmentationLoss calcula ``softmax`` una sola vez si alguna
  de las losses activas lo necesita (dice/lovasz).
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


_VALID_LOSSES = ("ce", "dice", "focal", "lovasz", "ohem_ce", "weighted_ce")
_NEEDS_PROBS  = ("dice", "lovasz")


# ╭───────────────────────────────────────────────────────────────────────╮
# │ Losses individuales                                                   │
# ╰───────────────────────────────────────────────────────────────────────╯

class CrossEntropy(nn.Module):
    """CE estándar píxel a píxel, con ignore_index y label smoothing opcional."""
    def __init__(self, ignore_index: int = 255, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets.long(), ignore_index=self.ignore_index,
                               label_smoothing=self.label_smoothing)


class DiceLoss(nn.Module):
    """
    Dice Loss = 1 − Dice score.

    Mide el solapamiento entre predicción y ground truth normalizado por el
    tamaño de ambos; da el mismo peso relativo a objetos pequeños y grandes
    (compensa el desbalanceo de clases). Recibe **probabilidades** (softmax),
    no logits.
    """
    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # probs:   (B, C, H, W) tras softmax
        # targets: (B, H, W) índices long
        num_classes  = probs.shape[1]
        targets      = targets.long()
        valid_mask   = targets != self.ignore_index
        targets_safe = targets.clone()
        targets_safe[~valid_mask] = 0                # one_hot peta con valores fuera de rango

        targets_oh = F.one_hot(targets_safe, num_classes).permute(0, 3, 1, 2).float()
        targets_oh = targets_oh * valid_mask.unsqueeze(1).float()   # anula los píxeles ignorados

        intersection = (probs * targets_oh).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets_oh.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss numéricamente estable, con label_smoothing opcional.

    Parte de la CE píxel a píxel y la multiplica por ``(1 − p_t)^gamma``, lo
    que reduce el peso de los píxeles ya bien clasificados → se enfoca en los
    difíciles. ``gamma = 0`` lo reduce a CE. Promedia sobre píxeles válidos.

    Con ``label_smoothing > 0`` se usa la CE suavizada como base
    (``F.cross_entropy(..., label_smoothing=ls, reduction='none')``) y
    ``pt = exp(-ce_smooth)`` — es la convención habitual en implementaciones
    de referencia de Focal+LS, no exactamente p_t pero suficientemente cercano.
    """
    def __init__(self, gamma: float = 2.0, ignore_index: int = 255,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)                       # p_t de la clase correcta (aprox. si hay smoothing)
        focal = (1.0 - pt) ** self.gamma * ce
        valid = (targets != self.ignore_index)
        n = valid.sum().clamp(min=1)
        return focal.sum() / n


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradiente de la extensión de Lovász (Berman 2018, Alg. 1).
    Sin operaciones in-place para que autograd no se queje."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if jaccard.numel() > 1:                  # diferencias finitas (sin in-place)
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]], dim=0)
    return jaccard


def _lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Lovász-Softmax (modo 'present') sobre tensores ya aplanados y filtrados."""
    if probs.numel() == 0:
        return probs.sum() * 0.0                # 0 conectado al grafo
    C = probs.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()              # presencia de la clase c
        if fg.sum() == 0:
            continue                            # 'present': se omiten clases ausentes en este batch
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    """
    Lovász-Softmax multi-clase (Berman et al. 2018) — surrogate diferenciable
    del IoU. Recibe **probabilidades** (softmax), no logits. Más lento que
    el resto (un sort por clase y por batch).
    """
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        # aplana y filtra píxeles ignorados
        probs_flat   = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1))
        targets_flat = targets.view(-1)
        valid        = targets_flat != self.ignore_index
        return _lovasz_softmax_flat(probs_flat[valid], targets_flat[valid])


class OhemCrossEntropy(nn.Module):
    """
    CE con Online Hard Example Mining: calcula la CE por píxel, descarta los
    ignorados y promedia solo sobre el ``top_k`` × N píxeles con más loss
    (los "difíciles"). ``top_k`` ∈ (0, 1]; default 0.25.
    Soporta ``label_smoothing`` (se aplica antes del topk).
    """
    def __init__(self, top_k: float = 0.25, ignore_index: int = 255,
                 label_smoothing: float = 0.0):
        super().__init__()
        if not 0.0 < top_k <= 1.0:
            raise ValueError(f"top_k debe estar en (0, 1], recibido {top_k}")
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        ce = F.cross_entropy(logits, targets, ignore_index=self.ignore_index,
                             label_smoothing=self.label_smoothing, reduction="none")
        ce_flat = ce.view(-1)
        valid   = (targets != self.ignore_index).view(-1)
        ce_valid = ce_flat[valid]
        if ce_valid.numel() == 0:
            return ce.sum() * 0.0
        k = max(1, int(round(self.top_k * ce_valid.numel())))
        topk, _ = ce_valid.topk(k)
        return topk.mean()


class WeightedCrossEntropy(nn.Module):
    """
    CE con un vector de pesos por clase (1 escalar por clase). Si
    ``class_weights`` es ``None`` se comporta como CE normal. El vector se
    mueve al device de los logits la primera vez que se llama a forward.
    """
    def __init__(self, class_weights: Optional[Union[torch.Tensor, Iterable[float]]] = None,
                 ignore_index: int = 255, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        if class_weights is None:
            self._weight: Optional[torch.Tensor] = None
        elif isinstance(class_weights, torch.Tensor):
            self._weight = class_weights.detach().float()
        else:
            self._weight = torch.tensor(list(class_weights), dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self._weight
        if w is not None and w.device != logits.device:
            w = w.to(logits.device)
            self._weight = w                      # cachea la versión movida
        return F.cross_entropy(logits, targets.long(), weight=w, ignore_index=self.ignore_index,
                               label_smoothing=self.label_smoothing)


# ╭───────────────────────────────────────────────────────────────────────╮
# │ Helper: class_weights "auto" (frecuencia inversa sobre el train set)  │
# ╰───────────────────────────────────────────────────────────────────────╯

def compute_class_weights(loader, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """Una pasada por el train_loader contando píxeles por clase y devuelve los
    pesos por **frecuencia inversa**, normalizados a media 1. Las clases ausentes
    reciben peso 1.0 (no penalizan ni inflan)."""
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in tqdm(loader, desc="class_weights (1 pasada train)", leave=False):
        masks = batch[1]
        flat  = masks.flatten().long()
        flat  = flat[(flat >= 0) & (flat < num_classes) & (flat != ignore_index)]
        if flat.numel() == 0:
            continue
        counts += torch.bincount(flat, minlength=num_classes).to(torch.float64)
    total = counts.sum().clamp(min=1.0)
    freq  = counts / total
    w     = torch.where(counts > 0,
                        1.0 / freq.clamp(min=1e-12),
                        torch.ones_like(freq, dtype=torch.float64))
    w = w / w.mean()
    return w.float()


# ╭───────────────────────────────────────────────────────────────────────╮
# │ SegmentationLoss combinada                                            │
# ╰───────────────────────────────────────────────────────────────────────╯

class SegmentationLoss(nn.Module):
    """
    Suma ponderada de las losses activas (peso > 0). Si alguna de las losses
    necesita probabilidades (dice/lovasz), se aplica ``softmax`` UNA sola vez
    por forward y se reusa.

    Parámetros
    ----------
    weights      Dict ``{nombre: peso}`` con los pesos de cada loss. Nombres
                 válidos: ``ce``, ``dice``, ``focal``, ``lovasz``, ``ohem_ce``,
                 ``weighted_ce``. Solo se instancian (y suman) las losses con
                 peso > 0. Claves desconocidas o dict vacío / todo a 0 →
                 ValueError.
    ignore_index Píxeles con este valor en el target se ignoran (default 255,
                 convención VOC).
    num_classes  Nº de clases del problema (necesario para ``weighted_ce``
                 ``"auto"``).
    focal_gamma  Exponente de la Focal Loss (default 2.0).
    ohem_top_k   Fracción de píxeles "más difíciles" en OHEM-CE (default 0.25).
    class_weights Pesos para ``weighted_ce``:
                 ``None`` → CE sin pesos.
                 ``"auto"`` → frecuencia inversa sobre el train set, cacheada
                              en ``checkpoints/class_weights_auto_<N>cls.pt``.
                 ``list[float]`` de longitud ``num_classes``.
    train_loader Solo se usa si ``class_weights == "auto"`` y no hay cache;
                 entonces se hace una pasada para calcular los pesos.
    cache_path   Ruta del cache de class_weights="auto". Si es None se usa
                 ``checkpoints/class_weights_auto_<num_classes>cls.pt``.
    """
    def __init__(
        self,
        weights: dict,
        ignore_index: int = 255,
        num_classes: int = 21,
        focal_gamma: float = 2.0,
        ohem_top_k: float = 0.25,
        class_weights=None,
        train_loader=None,
        cache_path: Optional[str] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        unknown = set(weights) - set(_VALID_LOSSES)
        if unknown:
            raise ValueError(f"Pesos con claves desconocidas: {sorted(unknown)}. "
                             f"Válidas: {list(_VALID_LOSSES)}")
        active = {k: float(v) for k, v in weights.items() if float(v) > 0.0}
        if not active:
            raise ValueError("SegmentationLoss: todos los pesos son 0 (o dict vacío). "
                             "Asigna peso > 0 al menos a una loss.")

        self.ignore_index    = ignore_index
        self.num_classes     = num_classes
        self.focal_gamma     = focal_gamma
        self.ohem_top_k      = ohem_top_k
        self.label_smoothing = label_smoothing
        self.class_weights_arg = class_weights        # se guarda para inspección/run_name
        self._weights     = active                    # dict ordenado de losses activas
        self._needs_probs = any(k in _NEEDS_PROBS for k in active)

        self.losses = nn.ModuleDict()
        if "ce" in active:
            self.losses["ce"] = CrossEntropy(ignore_index=ignore_index,
                                             label_smoothing=label_smoothing)
        if "dice" in active:
            self.losses["dice"] = DiceLoss(ignore_index=ignore_index)
        if "focal" in active:
            self.losses["focal"] = FocalLoss(gamma=focal_gamma, ignore_index=ignore_index,
                                             label_smoothing=label_smoothing)
        if "lovasz" in active:
            self.losses["lovasz"] = LovaszSoftmax(ignore_index=ignore_index)
        if "ohem_ce" in active:
            self.losses["ohem_ce"] = OhemCrossEntropy(top_k=ohem_top_k, ignore_index=ignore_index,
                                                     label_smoothing=label_smoothing)
        if "weighted_ce" in active:
            w = self._resolve_class_weights(class_weights, num_classes, ignore_index,
                                            train_loader, cache_path)
            self.losses["weighted_ce"] = WeightedCrossEntropy(
                class_weights=w, ignore_index=ignore_index,
                label_smoothing=label_smoothing,
            )

    # ── auxiliares ──────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_class_weights(class_weights, num_classes, ignore_index,
                                train_loader, cache_path):
        if class_weights is None:
            return None
        if isinstance(class_weights, (list, tuple)):
            t = torch.tensor(list(class_weights), dtype=torch.float32)
            if t.numel() != num_classes:
                raise ValueError(f"class_weights debe tener {num_classes} valores, no {t.numel()}")
            return t
        if isinstance(class_weights, torch.Tensor):
            if class_weights.numel() != num_classes:
                raise ValueError(f"class_weights debe tener {num_classes} valores, "
                                 f"no {class_weights.numel()}")
            return class_weights.detach().float()
        if isinstance(class_weights, str) and class_weights.lower() == "auto":
            path = cache_path or os.path.join(
                "checkpoints", f"class_weights_auto_{num_classes}cls.pt")
            if os.path.isfile(path):
                w = torch.load(path, map_location="cpu")
                if w.numel() != num_classes:
                    raise ValueError(f"Cache {path} tiene {w.numel()} clases "
                                     f"(esperaba {num_classes}). Bórralo y vuelve a entrenar.")
                print(f"[loss] class_weights='auto' cargados de cache: {path}")
                return w.float()
            if train_loader is None:
                raise ValueError(
                    f"class_weights='auto' sin cache en {path} y sin train_loader. "
                    "Ejecuta main.py al menos una vez para generar el cache, o pasa "
                    "una lista explícita.")
            print(f"[loss] class_weights='auto': calculando frecuencias en train "
                  f"(1 pasada, puede tardar varios minutos en COCO)...")
            w = compute_class_weights(train_loader, num_classes, ignore_index)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(w, path)
            print(f"[loss] class_weights guardados en {path}")
            return w
        raise ValueError(f"class_weights inválido: {class_weights!r}. "
                         "Usa None | 'auto' | lista de num_classes floats.")

    # ── forward ─────────────────────────────────────────────────────────────
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        probs = logits.softmax(dim=1) if self._needs_probs else None
        total: Optional[torch.Tensor] = None
        for name, w in self._weights.items():
            inp  = probs if name in _NEEDS_PROBS else logits
            term = w * self.losses[name](inp, targets)
            total = term if total is None else total + term
        return total       # garantizado no-None (al menos una loss activa)

    # ── repr / introspección ───────────────────────────────────────────────
    @property
    def weights(self) -> dict:
        """Dict de losses activas con sus pesos."""
        return dict(self._weights)

    def __repr__(self) -> str:
        parts = [f"{k}={v:g}" for k, v in self._weights.items()]
        extras = []
        if "focal" in self._weights and self.focal_gamma != 2.0:
            extras.append(f"focal_gamma={self.focal_gamma:g}")
        if "ohem_ce" in self._weights and self.ohem_top_k != 0.25:
            extras.append(f"ohem_top_k={self.ohem_top_k:g}")
        if "weighted_ce" in self._weights and self.class_weights_arg is not None:
            cw = self.class_weights_arg if isinstance(self.class_weights_arg, str) else "list"
            extras.append(f"class_weights={cw}")
        if self.label_smoothing > 0:
            extras.append(f"label_smoothing={self.label_smoothing:g}")
        head = "SegmentationLoss(" + ", ".join(parts) + ")"
        return head + (" [" + ", ".join(extras) + "]" if extras else "")


# ╭───────────────────────────────────────────────────────────────────────╮
# │ Tests de sanidad                                                      │
# ╰───────────────────────────────────────────────────────────────────────╯

if __name__ == "__main__":
    torch.manual_seed(0)
    NC, B, H, W = 5, 2, 32, 32
    logits = torch.randn(B, NC, H, W, requires_grad=True)
    masks  = torch.randint(0, NC, (B, H, W))
    masks[torch.rand(B, H, W) < 0.10] = 255           # ~10% píxeles ignorados

    pct_ign = (masks == 255).float().mean().item() * 100
    print("=" * 70)
    print(f"TEST DE SANIDAD — losses.py  (B={B}, C={NC}, H={H}, W={W}; ignore≈{pct_ign:.1f}%)")
    print("=" * 70)

    # 1) Cada loss aislada → escalar finito > 0 + backward sin errores
    print("\n[1] Cada loss aislada da escalar finito > 0:")
    individuales = [
        ("ce",          {}),
        ("dice",        {}),
        ("focal",       {}),
        ("lovasz",      {}),
        ("ohem_ce",     {}),
        ("weighted_ce", {"class_weights": [1.0] * NC}),     # con pesos = CE normal
    ]
    for name, extra in individuales:
        crit = SegmentationLoss(
            weights={name: 1.0}, ignore_index=255, num_classes=NC,
            focal_gamma=2.0, ohem_top_k=0.25, **extra,
        )
        loss = crit(logits, masks)
        assert torch.isfinite(loss), f"{name}: loss no finita = {loss}"
        assert loss.item() > 0, f"{name}: loss no positiva = {loss.item()}"
        loss.backward(retain_graph=True)
        logits.grad = None
        print(f"    {name:<12} loss = {loss.item():.4f}   {crit!r}")

    # 2) SegmentationLoss({'ce': 1.0}) ≈ nn.CrossEntropyLoss
    print("\n[2] SegmentationLoss({'ce':1.0}) coincide con nn.CrossEntropyLoss:")
    crit_ce = SegmentationLoss(weights={"ce": 1.0}, ignore_index=255, num_classes=NC)
    got     = crit_ce(logits, masks)
    ref     = nn.CrossEntropyLoss(ignore_index=255)(logits, masks.long())
    diff    = (got - ref).abs().item()
    assert torch.allclose(got, ref, atol=1e-6), f"diff demasiado grande: {diff}"
    print(f"    got = {got.item():.6f}   ref = {ref.item():.6f}   |Δ| = {diff:.2e}   OK")

    # 3) dict vacío o todos los pesos a 0 → ValueError
    print("\n[3] dict vacío o todos los pesos a 0 → ValueError:")
    for bad in ({}, {"ce": 0.0, "dice": 0.0, "focal": 0.0}):
        try:
            SegmentationLoss(weights=bad, ignore_index=255, num_classes=NC)
        except ValueError as e:
            print(f"    weights={bad}  →  ValueError OK")
            continue
        raise AssertionError(f"esperaba ValueError con weights={bad}")

    # 4) Claves desconocidas → ValueError
    print("\n[4] Claves desconocidas → ValueError:")
    try:
        SegmentationLoss(weights={"ce": 1.0, "foo": 2.0}, ignore_index=255, num_classes=NC)
    except ValueError as e:
        print(f"    weights={{'ce':1, 'foo':2}}  →  ValueError OK: {e}")
    else:
        raise AssertionError("esperaba ValueError con clave inválida")

    # 5) label_smoothing: CE con smoothing coincide con nn.CrossEntropyLoss(label_smoothing=...)
    print("\n[5] label_smoothing en CE coincide con nn.CrossEntropyLoss(label_smoothing=0.1):")
    crit_ls = SegmentationLoss(weights={"ce": 1.0}, ignore_index=255, num_classes=NC,
                               label_smoothing=0.1)
    got_ls  = crit_ls(logits, masks)
    ref_ls  = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)(logits, masks.long())
    diff    = (got_ls - ref_ls).abs().item()
    assert torch.allclose(got_ls, ref_ls, atol=1e-6), f"diff demasiado grande: {diff}"
    print(f"    got = {got_ls.item():.6f}   ref = {ref_ls.item():.6f}   |Δ| = {diff:.2e}   OK")
    # y que con focal+ls la loss cambia (deja de ser la misma que sin smoothing)
    crit_f_nols = SegmentationLoss(weights={"focal": 1.0}, ignore_index=255, num_classes=NC,
                                   focal_gamma=2.0)
    crit_f_ls   = SegmentationLoss(weights={"focal": 1.0}, ignore_index=255, num_classes=NC,
                                   focal_gamma=2.0, label_smoothing=0.1)
    f_nols = crit_f_nols(logits, masks)
    f_ls   = crit_f_ls(logits, masks)
    assert not torch.allclose(f_nols, f_ls, atol=1e-4), \
        f"focal con label_smoothing=0.1 debería diferir del focal sin smoothing"
    print(f"    focal sin LS = {f_nols.item():.4f}  con LS=0.1 = {f_ls.item():.4f}   OK (varían)")

    # 6) Combinada con varias activas — comprueba que es la suma de las partes
    print("\n[6] Combinada ≈ suma ponderada de las partes (focal 0.5 + dice 0.5):")
    crit_comb = SegmentationLoss(weights={"focal": 0.5, "dice": 0.5},
                                 ignore_index=255, num_classes=NC, focal_gamma=2.0)
    f_alone = SegmentationLoss(weights={"focal": 1.0},
                               ignore_index=255, num_classes=NC, focal_gamma=2.0)(logits, masks)
    d_alone = SegmentationLoss(weights={"dice": 1.0},
                               ignore_index=255, num_classes=NC)(logits, masks)
    expected = 0.5 * f_alone + 0.5 * d_alone
    got = crit_comb(logits, masks)
    diff = (got - expected).abs().item()
    assert torch.allclose(got, expected, atol=1e-6), f"diff demasiado grande: {diff}"
    print(f"    got = {got.item():.6f}   expected = {expected.item():.6f}   |Δ| = {diff:.2e}   OK")

    print("\n" + "=" * 70)
    print("TODO OK")
    print("=" * 70)
