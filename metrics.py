import torch


class SegmentationMetrics:
    """
    EXPLICACIÓ SIMPLE: Mètriques per a segmentació semàntica.

    Acumula la matriu de confusió (preds vs targets) píxel a píxel sobre tot
    el split de validació, i d'allà deriva totes les mètriques. Si els flags
    corresponents del Config (LOG_PIXEL_ACCURACY, LOG_F1_PER_CLASS,
    LOG_BOUNDARY_IOU) estan a True també calcula:
      - pixel_accuracy: % de píxels correctament classificats.
      - F1 per classe i mF1: mitjana sobre les classes presents.
      - boundary_IoU: IoU restringida als píxels de la franja de bord
        (kernel d'erosió 3×3); penalitza més els errors a les vores.

    El càlcul "estàndard" (mIoU + IoU per classe) sempre s'inclou.
    """
    def __init__(self, num_classes, ignore_index=255,
                 compute_pixel_accuracy=False, compute_f1=False,
                 compute_boundary_iou=False, boundary_kernel=3):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.compute_pixel_accuracy = compute_pixel_accuracy
        self.compute_f1             = compute_f1
        self.compute_boundary_iou   = compute_boundary_iou
        self.boundary_kernel        = boundary_kernel
        self.reinicialitzar()

    def reinicialitzar(self):
        """Buida la matriu de confusió (i la de bordes si està activa)."""
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )
        if self.compute_boundary_iou:
            self.boundary_cm = torch.zeros(
                self.num_classes, self.num_classes, dtype=torch.long
            )

    def actualitzar(self, preds, targets):
        """
        Acumula les prediccions d'aquest batch.
        preds:   (B, C, H, W) logits
        targets: (B, H, W)   índices long
        """
        preds_argmax = preds.argmax(dim=1)            # (B, H, W)
        preds_flat   = preds_argmax.view(-1).cpu()
        targets_flat = targets.view(-1).cpu().long()

        # filtrar ignore_index y valores fuera de rango
        valid = (targets_flat >= 0) & (targets_flat < self.num_classes) & \
                (targets_flat != self.ignore_index)
        pf = preds_flat[valid]
        tf = targets_flat[valid]

        if tf.numel() == 0:
            print("[WARNING] No valid pixels found! Saltando este batch.")
            return

        idx = self.num_classes * tf + pf
        cm  = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

        if self.compute_boundary_iou:
            self._update_boundary_cm(preds_argmax.cpu(), targets.cpu().long())

    def _update_boundary_cm(self, preds_argmax, targets):
        """Acumula la confusion matrix solo en píxeles del borde de cada clase."""
        # Borde = píxel cuya clase difiere de algún vecino en una ventana NxN.
        # Lo aproximamos comparando target con max-pool y -min-pool (1-px borde).
        # Para no inflar el borde alrededor de regiones ignoradas, descartamos
        # las posiciones cuya ventana toca cualquier píxel ignorado.
        import torch.nn.functional as F

        k = self.boundary_kernel
        pad = k // 2
        valid_mask = (targets != self.ignore_index)
        targets_safe = torch.where(valid_mask, targets, torch.zeros_like(targets))
        t_f = targets_safe.float().unsqueeze(1)        # (B, 1, H, W)

        # max-pool y -(-)min-pool sobre target → cualquier diferencia local marca borde
        t_max = F.max_pool2d(t_f, kernel_size=k, stride=1, padding=pad)
        t_min = -F.max_pool2d(-t_f, kernel_size=k, stride=1, padding=pad)
        # min-pool de valid_mask: =1 solo si TODA la ventana es válida; descarta
        # los bordes falsos junto a píxeles ignorados (que se rellenaron con 0).
        vm_f         = valid_mask.float().unsqueeze(1)
        valid_window = -F.max_pool2d(-vm_f, kernel_size=k, stride=1, padding=pad)
        boundary = (((t_max - t_min).squeeze(1) != 0) & valid_mask
                    & (valid_window.squeeze(1) > 0.5))

        if boundary.sum() == 0:
            return
        bf = boundary.view(-1)
        pf = preds_argmax.view(-1)[bf]
        tf = targets.view(-1)[bf]
        # filtrar ignore (por si acaso) y rango
        m = (tf >= 0) & (tf < self.num_classes) & (tf != self.ignore_index)
        pf, tf = pf[m], tf[m]
        if tf.numel() == 0:
            return
        idx = self.num_classes * tf + pf
        cm  = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.boundary_cm += cm.reshape(self.num_classes, self.num_classes)

    def calcular(self):
        """
        Retorna un dict con todas las métricas calculadas.
        Claves siempre presentes: mIoU, IoU_per_class.
        Claves opcionales (si los flags correspondientes están activos):
        pixel_accuracy, F1_per_class, mF1, boundary_mIoU, boundary_IoU_per_class.
        """
        cm   = self.confusion_matrix.float()
        total_pixels = cm.sum().item()

        if total_pixels == 0:
            print("[WARNING] Confusion matrix vacía. Métricas a 0.")
            empty = {
                "mIoU": 0.0,
                "IoU_per_class": [0.0] * self.num_classes,
            }
            if self.compute_pixel_accuracy:
                empty["pixel_accuracy"] = 0.0
            if self.compute_f1:
                empty["F1_per_class"] = [0.0] * self.num_classes
                empty["mF1"] = 0.0
            if self.compute_boundary_iou:
                empty["boundary_mIoU"] = 0.0
                empty["boundary_IoU_per_class"] = [0.0] * self.num_classes
            return empty

        diag = cm.diag()
        union = cm.sum(1) + cm.sum(0) - diag
        iou   = diag / (union + 1e-6)

        classes_present = cm.sum(1) > 0
        miou = iou[classes_present].mean().item() if classes_present.sum() > 0 else 0.0

        results = {
            "mIoU": miou,
            "IoU_per_class": iou.tolist(),
        }

        if self.compute_pixel_accuracy:
            results["pixel_accuracy"] = (diag.sum() / cm.sum().clamp(min=1)).item()

        if self.compute_f1:
            # F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c)
            tp = diag
            fp = cm.sum(0) - diag
            fn = cm.sum(1) - diag
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
            results["F1_per_class"] = f1.tolist()
            results["mF1"] = f1[classes_present].mean().item() if classes_present.sum() > 0 else 0.0

        if self.compute_boundary_iou:
            bcm   = self.boundary_cm.float()
            bdiag = bcm.diag()
            bunion = bcm.sum(1) + bcm.sum(0) - bdiag
            biou  = bdiag / (bunion + 1e-6)
            bpresent = bcm.sum(1) > 0
            bmiou = biou[bpresent].mean().item() if bpresent.sum() > 0 else 0.0
            results["boundary_mIoU"] = bmiou
            results["boundary_IoU_per_class"] = biou.tolist()

        return results
