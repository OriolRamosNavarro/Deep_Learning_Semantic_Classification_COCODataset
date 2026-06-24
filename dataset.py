import os
from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from classes import COCO_CAT_ID_TO_INDEX
from coco_filter import keep_area, masks_dirname, normalize_sizes


class SegmentationDataset(Dataset):
    """
    EXPLICACIÓ SIMPLE: Dataset personalitzat que carrega imatges i máscaras de segmentació.
    Les imatges han de ser .jpg i les máscaras .png al mateix ordre alfabètic.
    En cada accés, retorna una imatge i la seva màscara corresponent transformades.
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths  = sorted(glob(os.path.join(img_dir,  "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        assert len(self.img_paths) == len(self.mask_paths), \
            f"#imgs={len(self.img_paths)} != #masks={len(self.mask_paths)}"
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx])
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


class CocoSegmentation(Dataset):
    """Segmentación semántica sobre COCO derivada de anotaciones de instancias.

    Cada píxel recibe el ÍNDICE CONTIGUO de la categoría (1..80); el fondo queda en 0.
    Los category_id originales de COCO (1-90, no contiguos) se remapean con
    COCO_CAT_ID_TO_INDEX → 81 clases en total. Las instancias se superponen de
    mayor a menor área para que las pequeñas no queden tapadas.

    Estructura esperada en <root>:
        <root>/train2017/   <root>/val2017/
        <root>/annotations/instances_train2017.json
        <root>/annotations/instances_val2017.json
    """

    _SPLIT_DIR = {"train": "train2017", "val": "val2017"}

    def __init__(self, root: str, split: str = "train", transforms=None,
                 instance_sizes=None):
        from pycocotools.coco import COCO

        if split not in self._SPLIT_DIR:
            raise ValueError(f"split debe ser 'train' o 'val', no {split!r}")

        split_dir = self._SPLIT_DIR[split]
        ann_file  = os.path.join(root, "annotations", f"instances_{split_dir}.json")
        self.coco     = COCO(ann_file)
        self.img_dir  = os.path.join(root, split_dir)
        self.ids      = list(self.coco.imgs.keys())
        self.transforms = transforms
        # Tallas de instancia a conservar (se calcula una vez aquí, no por acceso).
        self._allowed = normalize_sizes(instance_sizes)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id   = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image    = Image.open(os.path.join(self.img_dir, img_info["file_name"])).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)

        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        # orden descendente de área: las instancias pequeñas se pintan encima
        for ann in sorted(anns, key=lambda a: a["area"], reverse=True):
            if not keep_area(ann["area"], self._allowed):
                continue  # instancia de talla filtrada → no se pinta (queda fondo)
            idx = COCO_CAT_ID_TO_INDEX.get(ann["category_id"])
            if idx is None:
                continue  # category_id no esperado (no debería pasar con instances_*.json)
            m = self.coco.annToMask(ann)
            mask[m > 0] = idx  # índice contiguo 1..80; 0 = fondo

        if self.transforms:
            image, mask = self.transforms(image, Image.fromarray(mask))
        return image, mask


class CocoSegmentationCached(Dataset):
    """COCO con máscaras pre-generadas a disco (ver tools/precompute_coco_masks.py).

    Mucho más rápido que CocoSegmentation porque NO llama a pycocotools.annToMask
    en cada acceso: solo lee un PNG ya generado. Reduce drásticamente el tiempo por
    epoch cuando el cuello de botella es la generación de máscaras.

    Las máscaras pre-generadas ya vienen remapeadas a índices contiguos 1..80
    (0 = fondo) — 81 clases. Si las generaste con una versión antigua del script
    (category_id crudos hasta 90), bórralas y vuelve a ejecutar el script.

    Las imágenes se leen de <root>/train2017|val2017/ y las máscaras de
    <masks_root>/masks_train2017|masks_val2017/. Si masks_root es None, se usa root
    (útil cuando el COCO es de solo lectura: pon masks_root en una carpeta tuya).
    """

    _SPLIT_DIR = {"train": "train2017", "val": "val2017"}

    def __init__(self, root: str, split: str = "train", transforms=None, masks_root=None,
                 instance_sizes=None):
        if split not in self._SPLIT_DIR:
            raise ValueError(f"split debe ser 'train' o 'val', no {split!r}")
        split_dir     = self._SPLIT_DIR[split]
        masks_base    = masks_root if masks_root else root
        self.img_dir  = os.path.join(root, split_dir)
        # Carpeta de máscaras según el filtro de tamaño: sin filtrar → 'masks_<split>'
        # (las de siempre); con filtro → carpeta con sufijo (debe haberse pre-generado
        # con tools/precompute_coco_masks.py --instance-sizes ...).
        mask_subdir   = masks_dirname(split_dir, instance_sizes)
        self.mask_dir = os.path.join(masks_base, mask_subdir)
        if not os.path.isdir(self.mask_dir):
            sizes_arg = " ".join(normalize_sizes(instance_sizes))
            raise FileNotFoundError(
                f"No existe {self.mask_dir}. Genera las máscaras primero:\n"
                f"  python tools/precompute_coco_masks.py --coco-root {root} "
                f"--masks-root {masks_base} --split {split} --instance-sizes {sizes_arg}"
            )
        self.mask_paths = sorted(glob(os.path.join(self.mask_dir, "*.png")))
        assert self.mask_paths, f"No se encontraron máscaras .png en {self.mask_dir}"
        self.transforms = transforms

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        mask_path = self.mask_paths[idx]
        stem      = os.path.splitext(os.path.basename(mask_path))[0]
        img_path  = os.path.join(self.img_dir, f"{stem}.jpg")
        image     = Image.open(img_path).convert("RGB")
        mask      = Image.open(mask_path)
        if self.transforms:
            image, mask = self.transforms(image, mask)
        return image, mask
