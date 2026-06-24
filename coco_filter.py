"""Filtrado de instancias de COCO por tamaño.

Convención COCO ESTÁNDAR (sobre `ann["area"]`, el área de la segmentación en px²):
    small  : area < 32² (= 1024)
    medium : 1024 ≤ area < 96² (= 9216)
    large  : area ≥ 9216

⚠️ MAPEO DE NOMBRES: en este proyecto usamos **'big'** en lugar del 'large' de COCO.
   `Config.INSTANCE_SIZES` se escribe con 'big'/'medium'/'small'; aquí 'big' se trata
   exactamente como el 'large' de COCO (area ≥ 96²). Se acepta 'large' como alias.

El filtrado es a NIVEL DE ANOTACIÓN: solo se pintan las instancias cuyo tamaño está
en INSTANCE_SIZES; las demás no se pintan → sus píxeles quedan como fondo (clase 0)
o como la instancia mayor que tengan debajo. No se descartan imágenes.

Las funciones de este módulo se usan en el punto de GENERACIÓN de la máscara
(tools/precompute_coco_masks.py y dataset.CocoSegmentation), no en cada acceso del
DataLoader: con máscaras cacheadas el filtrado ya viene "horneado" en disco.
"""

SMALL_MAX  = 32 ** 2   # 1024  → área por debajo = 'small'
MEDIUM_MAX = 96 ** 2   # 9216  → área por debajo (y ≥ SMALL_MAX) = 'medium'; ≥ = 'big'

# Orden fijo (de mayor a menor) para nombres de carpeta deterministas.
ALL_SIZES = ("big", "medium", "small")   # 'big' == 'large' de COCO


def size_of(area: float) -> str:
    """Clasifica un área (px²) en 'small' | 'medium' | 'big' (large→big)."""
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "big"


def normalize_sizes(instance_sizes) -> set:
    """Valida y normaliza INSTANCE_SIZES → set en minúsculas.

    None / vacío / 'all' → las tres tallas (sin filtrar). 'large' se acepta como
    alias de 'big'. Lanza ValueError si hay alguna talla desconocida.
    """
    if not instance_sizes or instance_sizes == "all":
        return set(ALL_SIZES)
    sizes = {str(s).strip().lower() for s in instance_sizes}
    if "large" in sizes:                       # alias tolerado
        sizes.discard("large")
        sizes.add("big")
    invalid = sizes - set(ALL_SIZES)
    if invalid:
        raise ValueError(
            f"INSTANCE_SIZES inválido: {sorted(invalid)}. "
            f"Usa un subconjunto de {list(ALL_SIZES)} (o 'large' como alias de 'big')."
        )
    if not sizes:
        return set(ALL_SIZES)
    return sizes


def is_filtering(instance_sizes) -> bool:
    """True si el filtrado descarta algo (subconjunto ESTRICTO de las 3 tallas)."""
    return normalize_sizes(instance_sizes) != set(ALL_SIZES)


def keep_area(area: float, allowed: set) -> bool:
    """True si una instancia de esa área debe conservarse (su talla está en `allowed`)."""
    return size_of(area) in allowed


def masks_dirname(split_dir: str, instance_sizes) -> str:
    """Nombre de la carpeta de máscaras según el filtro.

    - Sin filtrar (las 3 tallas) → 'masks_<split>'  ← idéntico al pipeline actual,
      así NO se regeneran ni se mezclan máscaras (garantiza el sanity check).
    - Con filtro → sufijo determinista 'masks_<split>__<tallas>' (orden big-medium-small),
      para que cada configuración de tamaños tenga su propia carpeta y no colisione.
    """
    base = f"masks_{split_dir}"
    if not is_filtering(instance_sizes):
        return base
    allowed = normalize_sizes(instance_sizes)
    suffix = "-".join(s for s in ALL_SIZES if s in allowed)
    return f"{base}__{suffix}"
