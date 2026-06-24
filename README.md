# Segmentación Semántica con Imágenes Naturales — XNAP Project 05

Proyecto del curso *Xarxes Neuronals i Aprenentatge Profund* (Grau d'Enginyeria de Dades, UAB, 2026) — segmentación semántica píxel a píxel sobre imágenes naturales (COCO), con encoder **ResNet152 preentrenado en ImageNet**. Se comparan dos arquitecturas de decoder, **U-Net** y **DeepLabV3+**, sobre las 3 mejores funciones de pérdida de un sweep previo.

> **Objetivo**: dada una imagen RGB, predecir un mapa `(H × W)` donde cada píxel tiene asignada una clase semántica.

> **Modelo final (ganador): DeepLabV3+ (ResNet152) con loss `0.33·CE + 0.33·Focal + 0.33·Lovász` → mIoU 0.6039, y 0.6283 con TTA.** Es la config por defecto de `config.py`.

---

## 1. Tarea y datos

- **Tarea**: segmentación semántica multiclase (una clase por píxel).
- **Dataset principal**: [COCO](https://cocodataset.org), redimensionado a **256 × 256**.
- **Dataset inicial (baseline rápido)**: [PASCAL VOC 2012 Segmentation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), 21 clases (incluye fondo + `ignore_index=255`). Se descarga directamente con `torchvision.datasets.VOCSegmentation`.

Empezamos por VOC porque su tamaño es manejable (~11K imágenes), está soportado nativamente por `torchvision` y permite tener un baseline funcional rápido antes de migrar a COCO.

---

## 2. Arquitectura

El encoder (ResNet152 preentrenado) es común; se conmuta el decoder con `Config.DECODER_TYPE`:

- **`"deeplabv3plus"`** (por defecto, **ganador**): encoder dilatado (output stride 16) + módulo **ASPP** + decoder que fusiona features de bajo nivel. `models/deeplab.py`.
- **`"unet"`**: skip connections en 4 escalas, decoder con `ConvTranspose2d` + concatenación + 2× `Conv-BN-ReLU`. `models/unet.py`.

Ambos terminan en un head `Conv 1×1 → num_classes` + interpolación bilinear al tamaño de entrada. Congelación del encoder: `layer0` y `layer1` congeladas, `layer2/3/4` fine-tuned (mejor mIoU empírico).

Esquema de la U-Net (alternativa):

```
Input (B, 3, 256, 256)
    │
    ▼
[Encoder ResNet50 preentrenado]
    │  layer0 (1/4)   ───────┐
    │  layer1 (1/4)   ─────┐ │
    │  layer2 (1/8)   ───┐ │ │
    │  layer3 (1/16)  ─┐ │ │ │
    ▼                  │ │ │ │
[Bottleneck layer4 (1/32, 2048ch)]
    │                  │ │ │ │
    ▼                  │ │ │ │
[Decoder: 4 × (Up + skip + 2× Conv-BN-ReLU)]
    │  ◄───────────────┘ │ │ │
    │  ◄─────────────────┘ │ │
    │  ◄───────────────────┘ │
    │  ◄─────────────────────┘
    ▼
[Conv 1×1 → num_classes]  →  bilinear → (B, num_classes, 256, 256)
```

---

## 3. Pérdida y métrica

- **Pérdida combinada (ganadora): `0.33·CE + 0.33·Focal + 0.33·Lovász`**
  - Cross-Entropy con `ignore_index=255` — bien condicionada y robusta.
  - Focal Loss — baja el peso de los píxeles fáciles (fondo dominante).
  - Lovász-Softmax — optimiza directamente el IoU; clave para la métrica.
  - La loss es configurable: cualquier combinación de CE / Dice / Focal / Lovász / OHEM-CE / Weighted-CE vía `config.py` o flags `--*-weight`.
- **Métrica principal**: **mIoU** (mean Intersection over Union) calculado sobre la confusion matrix acumulada del epoch completo (no promediando por batch). También se reporta **IoU por clase** y, durante el entrenamiento, `train_mIoU` vs `val_mIoU` para vigilar el overfitting.

> Nota: el `val_loss` **no** es comparable entre losses distintas (escalas distintas); para comparar modelos usa siempre el **mIoU**.

---

## 3b. Resultados de la comparativa final (COCO val, ResNet152, 50 epochs)

| Arquitectura | Loss | mIoU |
|---|---|---|
| **DeepLabV3+** | **ce+focal+lovasz** | **0.6039**  → **0.6283 con TTA** |
| DeepLabV3+ | focal+lovasz | 0.5951 |
| DeepLabV3+ | focal | 0.5894 |
| U-Net | ce+focal+lovasz | 0.5859 |
| U-Net | focal+lovasz | 0.5801 |
| U-Net | focal | 0.5698 |

**Conclusiones:** DeepLabV3+ supera a U-Net en las 6 combinaciones; la loss `ce+focal+lovasz` es la mejor en ambas arquitecturas. **TTA** (multiescala 0.75/1.0/1.25 + hflip) añade **+2.4 pp** sin reentrenar.

---

## 4. Estructura del repositorio

```
.
├── README.md            # este archivo
├── LICENSE
├── environment.yml      # entorno conda
├── main.py              # punto de entrada — orquesta todo
├── config.py            # hiperparámetros
├── dataset.py           # SegmentationDataset (modo manual con img_dir + mask_dir)
├── transforms.py        # transform sincronizado imagen-máscara
├── losses.py            # DiceLoss + SegmentationLoss combinada
├── metrics.py           # SegmentationMetrics (confusion matrix → mIoU)
├── engine.py            # train_one_epoch() + validate()
├── models/
│   ├── __init__.py
│   └── unet.py          # Encoder ResNet50 + DecoderBlock + UNet
├── docs/
│   └── informe_seguimiento_1.pdf
└── test/                # checks de GitHub Classroom (no tocar)
```

---

## 5. Instalación

```bash
conda activate grupo-5
```

Para tracking de experimentos (Wandb):

```bash
wandb login
```

---

## 6. Cómo reproducir

### Sanity check — overfit con 5 imágenes
Demuestra que el pipeline aprende:

```bash
python main.py --overfit 5 --epochs 30 --no-wandb
```

La loss debe bajar a casi cero. Si no, hay un bug en el pipeline.

### Entrenar el modelo ganador (config por defecto)
`config.py` ya trae la config ganadora (DeepLabV3+ + ce/focal/lovász), así que:

```bash
python main.py --data-root /home/datasets/coco --epochs 50
```

### Evaluar con TTA (mejor resultado)
```bash
python evaluate.py --ckpt checkpoints/best.pt --data-root /home/datasets/coco --tta --no-figure
```
TTA (multiescala + hflip) sube el mIoU sin reentrenar (0.6039 → 0.6283).

### Argumentos disponibles (selección)
| Flag | Descripción |
|---|---|
| `--data-root` | Carpeta del dataset (COCO en `/home/datasets/coco`) |
| `--epochs` | Sobrescribe `Config.EPOCHS` |
| `--overfit N` | Entrena/valida sobre las primeras `N` imágenes (sanity check) |
| `--ce-weight / --focal-weight / --lovasz-weight / ...` | Pesos de cada loss |
| `--wandb-project / --wandb-run-name / --ckpt-dir` | Proyecto, nombre y carpeta de checkpoints |
| `--no-wandb` / `--wandb-offline` | Logging de Wandb |
| `--tta` (en `evaluate.py`) | Test-Time Augmentation |

Los checkpoints se guardan en `<ckpt-dir>/best.pt` (mejor mIoU de validación).

---

## 7. Decisiones de diseño

| Decisión | Razón |
|---|---|
| DeepLabV3+ como decoder (vs U-Net) | Gana a U-Net en las 6 combinaciones probadas (ASPP capta contexto multiescala) |
| ResNet152 como encoder (no VGG) | Sin FC al final, skip connections residuales internas, mejor flujo de gradiente |
| Pesos ImageNet | Transfer learning aprovecha features de bajo/medio nivel |
| Congelar `layer0/1`, fine-tune `layer2/3/4` | Reutiliza bordes/texturas de ImageNet; reentrena formas/semántica para COCO |
| `LR_ENCODER` (1e-5) ≪ `LR_DECODER` (1e-4) | Encoder preentrenado: bajar LR evita destruir pesos. Decoder se entrena desde cero |
| CE + Focal + Lovász | Mejor loss del sweep; Lovász optimiza el IoU directamente |
| TTA en evaluación | +2.4 pp de mIoU sin reentrenar |
| `IMG_SIZE = 256` | Lo pide el enunciado del proyecto |
| `ignore_index = 255` | Convención de VOC para píxeles de borde no etiquetados |

---

## 8. Estado actual

- [x] Dataset (VOC + COCO con máscaras pre-generadas) y transform sincronizado
- [x] U-Net y DeepLabV3+ con ResNet152 preentrenado (conmutables por `DECODER_TYPE`)
- [x] Loss combinada configurable (CE / Dice / Focal / Lovász / OHEM-CE / Weighted-CE)
- [x] Métricas (mIoU + IoU por clase, train_mIoU vs val_mIoU)
- [x] Training loop + validación + EMA + warmup/cosine + grad clip
- [x] `main.py` con CLI (overfit, wandb, epochs, pesos de loss, ckpt-dir)
- [x] Wandb integrado
- [x] Migración a COCO (81 clases)
- [x] Sweep de 31 losses + comparativa final U-Net vs DeepLabV3+
- [x] TTA en evaluación → **mejor mIoU 0.6283**

---

## 9. Referencias

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- COCO dataset: <https://cocodataset.org>
- PASCAL VOC 2012: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>
