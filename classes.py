"""
EXPLICACIÓ SIMPLE: Noms i colors de les classes dels datasets suportats (VOC2012 i COCO).

VOC2012 → 21 classes (20 objectes + fons), índexs contigus 0..20.
COCO     → 81 classes (80 objectes "thing" + fons). Els category_id originals de COCO
           van de 1 a 90 amb forats; aquí es remapegen a índexs contigus 1..80 (0 = fons).
           Aquest remapeig l'apliquen dataset.py (CocoSegmentation) i
           tools/precompute_coco_masks.py quan generen les màscares.

Els colormaps serveixen per pintar les màscares (índex de classe → color RGB) en evaluate.py.
"""

# ───────────────────────────────────────── VOC2012 ─────────────────────────────────────────
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
assert len(VOC_CLASSES) == 21

VOC_COLORMAP = [
    (0, 0, 0),       (128, 0, 0),     (0, 128, 0),    (128, 128, 0),
    (0, 0, 128),     (128, 0, 128),   (0, 128, 128),  (128, 128, 128),
    (64, 0, 0),      (192, 0, 0),     (64, 128, 0),   (192, 128, 0),
    (64, 0, 128),    (192, 0, 128),   (64, 128, 128), (192, 128, 128),
    (0, 64, 0),      (128, 64, 0),    (0, 192, 0),    (128, 192, 0),
    (0, 64, 128),
]
assert len(VOC_COLORMAP) == 21


def _gen_colormap(n: int):
    """Genera n colores distintos con el mismo algoritmo de bits que la paleta de VOC."""
    def bit(val, idx):
        return (val >> idx) & 1

    colormap = []
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bit(c, 0) << (7 - j)
            g |= bit(c, 1) << (7 - j)
            b |= bit(c, 2) << (7 - j)
            c >>= 3
        colormap.append((r, g, b))
    return colormap


# ─────────────────────────────────────────── COCO ──────────────────────────────────────────
# 80 categorías "thing" de COCO, en orden de su category_id original.
COCO_THING_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
assert len(COCO_THING_CLASSES) == 80

# Índice 0 = fondo, índices 1..80 = las 80 categorías de COCO (contiguas).
COCO_CLASSES = ("background",) + COCO_THING_CLASSES
assert len(COCO_CLASSES) == 81

# category_id original de COCO (1-90, NO contiguos) → índice contiguo (1..80); 0 = fondo.
_COCO_CAT_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
)
assert len(_COCO_CAT_IDS) == 80
COCO_CAT_ID_TO_INDEX = {cat_id: i + 1 for i, cat_id in enumerate(_COCO_CAT_IDS)}

COCO_COLORMAP = _gen_colormap(81)
assert len(COCO_COLORMAP) == 81


# ─────────────────────────────────────── helpers ───────────────────────────────────────────
def get_classes(dataset: str) -> tuple:
    d = dataset.upper()
    if d in ("VOC", "VOC2012"):
        return VOC_CLASSES
    if d == "COCO":
        return COCO_CLASSES
    raise ValueError(f"Dataset desconocido: {dataset!r}. Usa 'VOC' o 'COCO'.")


def get_colormap(dataset: str) -> list:
    d = dataset.upper()
    if d in ("VOC", "VOC2012"):
        return VOC_COLORMAP
    if d == "COCO":
        return COCO_COLORMAP
    raise ValueError(f"Dataset desconocido: {dataset!r}. Usa 'VOC' o 'COCO'.")
