import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from PIL import Image, ImageFilter


class PairedTransform:
    """
    EXPLICACIÓ SIMPLE: Transformació sincronitzada imatge-màscara per a segmentació.
    Aplica EXACTAMENT els mateixos canvis geomètrics a la imatge i a la màscara
    (resize, flip, rotació, afí) perquè els píxels segueixin alineats amb les classes.
    Els canvis de color (brillantor, contrast, to...) i el blur només s'apliquen a la imatge.
    La imatge es normalitza amb les estadístiques d'ImageNet; la màscara es manté com
    a LongTensor amb els índexs de classe intactes.

    Todos los hiperparámetros de las augmentations se leen del Config (sección 2,
    AUG_*). Para sobrescribirlos puntualmente se pueden pasar como argumentos al
    constructor; los que no se pasen, se leen del Config.

    Augmentaciones de entrenamiento:
      - Random scale (RANGE) + random crop a img_size x img_size
      - Random horizontal flip            (p = AUG_HFLIP_P)
      - Random vertical flip              (p = AUG_VFLIP_P; típicamente 0)
      - Random rotation  ±AUG_ROTATION_DEG (p = AUG_ROTATION_P)
      - Random affine shear ±AUG_AFFINE_SHEAR_DEG (p = AUG_AFFINE_P)
      - Random brightness / contrast      (p = AUG_BC_P)
      - Random hue / saturation           (p = AUG_HS_P)
      - Random gamma                      (p = AUG_GAMMA_P)
      - Random Gaussian blur              (p = AUG_BLUR_P)
    En validación solo se aplica el resize + normalización.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # Defaults usados si no se pasa ni cfg ni el parámetro individual.
    # Coinciden con los valores que estaban hardcoded antes del refactor.
    _DEFAULTS = dict(
        hflip_p=0.5,
        vflip_p=0.0,
        scale_range=(0.5, 2.0),
        rotation_deg=15.0,
        rotation_p=0.5,
        affine_shear_deg=10.0,
        affine_p=0.3,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        bc_p=0.5,
        hue_range=(-0.1, 0.1),
        saturation_range=(0.8, 1.2),
        hs_p=0.5,
        gamma_range=(0.8, 1.2),
        gamma_p=0.25,
        blur_radius_range=(0.5, 1.5),
        blur_p=0.25,
        ignore_index=255,
    )

    def __init__(self, img_size=256, train=True, cfg=None, **overrides):
        """
        Args
        ----
        img_size: tamaño final (cuadrado) de imagen y máscara.
        train:    si False, NO aplica augmentation (solo resize + normalize).
        cfg:      objeto Config (o cualquiera con atributos AUG_*); de él se leen
                  los hiperparámetros. Si es None se usan los _DEFAULTS.
        **overrides: cualquier hiperparámetro del _DEFAULTS pasado aquí
                     sobrescribe al valor leído del cfg (útil para tests).
        """
        self.img_size = img_size
        self.train    = train
        # 1) defaults → 2) cfg → 3) overrides explícitos
        params = dict(self._DEFAULTS)
        if cfg is not None:
            params.update(self._params_from_cfg(cfg))
        params.update(overrides)
        for k, v in params.items():
            setattr(self, k, v)

    @staticmethod
    def _params_from_cfg(cfg):
        """Lee los AUG_* y IGNORE_INDEX del Config, traduciéndolos al naming interno."""
        mapping = {
            "AUG_HFLIP_P":           "hflip_p",
            "AUG_VFLIP_P":           "vflip_p",
            "AUG_RANDOM_SCALE_RANGE": "scale_range",
            "AUG_ROTATION_DEG":      "rotation_deg",
            "AUG_ROTATION_P":        "rotation_p",
            "AUG_AFFINE_SHEAR_DEG":  "affine_shear_deg",
            "AUG_AFFINE_P":          "affine_p",
            "AUG_BRIGHTNESS_RANGE":  "brightness_range",
            "AUG_CONTRAST_RANGE":    "contrast_range",
            "AUG_BC_P":              "bc_p",
            "AUG_HUE_RANGE":         "hue_range",
            "AUG_SATURATION_RANGE":  "saturation_range",
            "AUG_HS_P":              "hs_p",
            "AUG_GAMMA_RANGE":       "gamma_range",
            "AUG_GAMMA_P":           "gamma_p",
            "AUG_BLUR_RADIUS_RANGE": "blur_radius_range",
            "AUG_BLUR_P":            "blur_p",
            "IGNORE_INDEX":          "ignore_index",
        }
        return {dst: getattr(cfg, src) for src, dst in mapping.items() if hasattr(cfg, src)}

    def __call__(self, image, mask):
        if self.train:
            # ── 1) Random scale + random crop a img_size x img_size ────────────
            lo, hi = self.scale_range
            scale_factor = random.uniform(lo, hi)
            w, h = image.size
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            image = TF.resize(image, (new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  (new_h, new_w), interpolation=T.InterpolationMode.NEAREST)

            pad_h = max(self.img_size - new_h, 0)
            pad_w = max(self.img_size - new_w, 0)
            if pad_h > 0 or pad_w > 0:
                image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
                mask  = TF.pad(mask,  [0, 0, pad_w, pad_h], fill=self.ignore_index)

            cur_w, cur_h = image.size
            top  = random.randint(0, cur_h - self.img_size)
            left = random.randint(0, cur_w - self.img_size)
            image = TF.crop(image, top, left, self.img_size, self.img_size)
            mask  = TF.crop(mask,  top, left, self.img_size, self.img_size)

            # ── 2) Geometría (sincronizada imagen + máscara) ───────────────────
            if self.hflip_p > 0 and random.random() < self.hflip_p:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            if self.vflip_p > 0 and random.random() < self.vflip_p:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            if self.rotation_p > 0 and random.random() < self.rotation_p:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                image = TF.rotate(image, angle, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=T.InterpolationMode.NEAREST,
                                  fill=self.ignore_index)

            if self.affine_p > 0 and random.random() < self.affine_p:
                shear = (random.uniform(-self.affine_shear_deg, self.affine_shear_deg),
                         random.uniform(-self.affine_shear_deg, self.affine_shear_deg))
                image = TF.affine(image, angle=0, translate=(0, 0), scale=1.0,
                                  shear=shear, interpolation=T.InterpolationMode.BILINEAR)
                mask  = TF.affine(mask,  angle=0, translate=(0, 0), scale=1.0,
                                  shear=shear, interpolation=T.InterpolationMode.NEAREST,
                                  fill=self.ignore_index)

            # ── 3) Color (solo imagen) ─────────────────────────────────────────
            if self.bc_p > 0 and random.random() < self.bc_p:
                image = TF.adjust_brightness(image, random.uniform(*self.brightness_range))
                image = TF.adjust_contrast(image,   random.uniform(*self.contrast_range))

            if self.hs_p > 0 and random.random() < self.hs_p:
                image = TF.adjust_hue(image,        random.uniform(*self.hue_range))
                image = TF.adjust_saturation(image, random.uniform(*self.saturation_range))

            if self.gamma_p > 0 and random.random() < self.gamma_p:
                gamma = random.uniform(*self.gamma_range)
                arr   = np.asarray(image, dtype=np.float32) / 255.0
                arr   = np.power(arr, gamma)
                image = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))

            if self.blur_p > 0 and random.random() < self.blur_p:
                image = image.filter(ImageFilter.GaussianBlur(
                    radius=random.uniform(*self.blur_radius_range)))
        else:
            # Validación: resize determinista (sin augmentation)
            size  = (self.img_size, self.img_size)
            image = TF.resize(image, size, interpolation=T.InterpolationMode.BILINEAR)
            mask  = TF.resize(mask,  size, interpolation=T.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, self.IMAGENET_MEAN, self.IMAGENET_STD)
        mask  = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask
