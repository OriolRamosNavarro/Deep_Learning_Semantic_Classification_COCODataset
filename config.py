"""
Configuración centralizada del proyecto.

Convención de marcas en este archivo:
    ✅  ACTIVO          → el código actual lee y aplica este parámetro
    ⚠️  PARCIAL         → el parámetro existe pero solo cubre un subconjunto
    🔧  FUTURO          → placeholder; el código aún NO lo aplica.
                          Cambiar el valor no tiene efecto hasta que se
                          implemente la técnica correspondiente.

Cualquier parámetro de este archivo puede sobrescribirse desde la CLI con
un flag análogo (ej: BATCH_SIZE → --batch-size). Los flags que ya existen
son los de la sección "PÉRDIDA COMBINADA" (--ce-weight, --dice-weight, ...);
el resto se irán añadiendo a main.py según se vayan necesitando.
"""

class Config:
    """Todos los hiperparámetros del modelo y del entrenamiento."""

    # ═══════════════════════════════════════════════════════════════════════
    # 1. DATOS
    # ═══════════════════════════════════════════════════════════════════════
    # Dataset y formato de entrada.
    DATASET     = "COCO"          # ✅ "COCO" | "VOC" | "VOC2012"
    NUM_CLASSES = 81              # ✅ COCO: 81 (fondo + 80) | VOC: 21
    IMG_SIZE    = 256             # ✅ enunciado pide 256
    IGNORE_INDEX = 255            # ✅ convención VOC para píxeles de borde

    # Carpeta donde están / se guardan las máscaras pre-generadas de COCO.
    # None → usa la misma carpeta de --data-root.
    MASKS_ROOT = "/home/edxnG05/coco_masks"   # ✅

    # DataLoader.
    BATCH_SIZE      = 32           # ✅
    NUM_WORKERS     = 8            # ✅
    PREFETCH_FACTOR = 4            # ✅ batches que precarga cada worker

    # Subset de clases COCO (entrenar con solo N de las 80).
    # None  → todas las 80; lista de category_id COCO → solo esas;
    # los píxeles de las demás se marcan como IGNORE_INDEX.
    COCO_KEEP_CLASSES = None       # 🔧 FUTURO

    # Filtra instancias de COCO por TAMAÑO (convención COCO por área de la segmentación):
    #   small: area < 32²(=1024) | medium: 1024 ≤ area < 96²(=9216) | big(=large): area ≥ 9216
    # 'big' es nuestro nombre para el 'large' de COCO (ver coco_filter.py). Solo se conservan
    # las instancias cuya talla esté en la lista; las demás pasan a fondo (filtrado a nivel de
    # anotación). La lista COMPLETA = sin filtrar (idéntico al pipeline actual).
    # ⚠️ Al cambiarla hay que regenerar máscaras: las filtradas se guardan en una carpeta
    #    con sufijo (p.ej. masks_train2017__big-medium) — ver tools/precompute_coco_masks.py.
    INSTANCE_SIZES = ["big", "medium"]   # ✅ experimento combinado: sin instancias 'small'

    # ═══════════════════════════════════════════════════════════════════════
    # 2. DATA AUGMENTATION (transforms.py)
    # ═══════════════════════════════════════════════════════════════════════
    # — Geométricas (sincronizadas imagen + máscara) —
    AUG_HFLIP_P              = 0.5            # ✅ probabilidad de flip horizontal
    AUG_VFLIP_P              = 0.0            # ✅ probabilidad de flip vertical (no recomendado)
    AUG_RANDOM_SCALE_RANGE   = (0.5, 2.0)     # ✅ rango de escalado aleatorio (lo + crop)
    AUG_ROTATION_DEG         = 15             # ✅ ±grados máximos de rotación
    AUG_ROTATION_P           = 0.5            # ✅
    AUG_AFFINE_SHEAR_DEG     = 10             # ✅ ±grados máximos de shear
    AUG_AFFINE_P             = 0.3            # ✅

    # — Color (solo imagen) —
    AUG_BRIGHTNESS_RANGE     = (0.8, 1.2)     # ✅
    AUG_CONTRAST_RANGE       = (0.8, 1.2)     # ✅
    AUG_BC_P                 = 0.5            # ✅ probabilidad conjunta brillo+contraste
    AUG_HUE_RANGE            = (-0.1, 0.1)    # ✅
    AUG_SATURATION_RANGE     = (0.8, 1.2)     # ✅
    AUG_HS_P                 = 0.5            # ✅ probabilidad conjunta hue+saturation
    AUG_GAMMA_RANGE          = (0.8, 1.2)     # ✅
    AUG_GAMMA_P              = 0.25           # ✅
    AUG_BLUR_RADIUS_RANGE    = (0.5, 1.5)     # ✅
    AUG_BLUR_P               = 0.25           # ✅

    # — Avanzadas —
    AUG_CUTMIX_P             = 0.0            # 🔧 FUTURO: CutMix entre 2 imágenes
    AUG_COPYPASTE_P          = 0.0            # 🔧 FUTURO: pegar instancias entre imágenes
    AUG_MIXUP_ALPHA          = 0.0            # 🔧 FUTURO: alpha de Mixup (0 = off)
    AUG_RANDOM_ERASING_P     = 0.0            # 🔧 FUTURO

    # ═══════════════════════════════════════════════════════════════════════
    # 3. MODELO
    # ═══════════════════════════════════════════════════════════════════════
    # — Encoder —
    BACKBONE   = "resnet152"       # ✅ resnet18 | resnet34 | resnet50 | resnet101 | resnet152
    PRETRAINED = True              # ✅ pesos ImageNet

    # Congelación capa a capa del encoder (True = congelada).
    FREEZE_LAYER0 = True           # ✅ conv inicial + maxpool (bordes — congelada)
    FREEZE_LAYER1 = True           # ✅ primer bloque ResNet (texturas — congelada)
    FREEZE_LAYER2 = False          # ✅ segundo bloque (formas — ENTRENABLE)
    FREEZE_LAYER3 = False          # ✅ tercer bloque (partes — ENTRENABLE)
    FREEZE_LAYER4 = False          # ✅ cuarto bloque (semántico — ENTRENABLE; mejor mIoU empírico)

    # — Decoder —
    DECODER_TYPE              = "deeplabv3plus"  # ✅ "unet" | "deeplabv3plus"  (rama: probamos DeepLab)
    DECODER_USE_BILINEAR_UP   = False         # 🔧 False=ConvTranspose2d (actual), True=bilinear+conv
    DECODER_DROPOUT           = 0.0           # ✅ dropout 2D al final del decoder (UNet y DeepLab)

    # — DeepLabV3+ específico (solo se leen si DECODER_TYPE == "deeplabv3plus") —
    DEEPLAB_OUTPUT_STRIDE  = 16     # ✅ 16 (default, recomendado) | 8 (más VRAM, ~+1pp mIoU)
    DEEPLAB_ASPP_OUT       = 256    # ✅ canales de salida del módulo ASPP (256 estándar)
    DEEPLAB_DECODER_LOW_CH = 48     # ✅ canales de proyección de las low-level features (48 estándar)
    DEEPLAB_ASPP_DROPOUT   = 0.1    # ✅ dropout dentro del módulo ASPP

    # — Módulos extra del modelo —
    # Fusión del decoder DeepLabV3+: False = concat ASPP+low-level (original);
    # True = CrossAttentionFusion (atención cruzada low←ASPP). Solo aplica a DeepLabV3+.
    # (Distinto de USE_ATTENTION_GATES, que sería para la U-Net.)
    USE_ATTENTION       = False     # ✅ solo DeepLabV3+ (experimento combinado: attention ON)

    USE_ATTENTION_GATES = False    # 🔧 Attention gates en skip connections (Attention U-Net)
    USE_SE_BLOCKS       = False    # 🔧 Squeeze-Excitation blocks
    USE_CBAM            = False    # 🔧 Convolutional Block Attention Module
    USE_ASPP            = False    # 🔧 Atrous Spatial Pyramid Pooling (estilo DeepLab)
    USE_AUX_HEAD        = False    # 🔧 cabeza auxiliar para deep supervision
    AUX_LOSS_WEIGHT     = 0.4      # 🔧 peso de la loss auxiliar

    # ═══════════════════════════════════════════════════════════════════════
    # 4. ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════════
    EPOCHS        = 50             # ✅
    WARMUP_EPOCHS = 2              # ✅ warmup lineal del LR

    # — Optimizer —
    OPTIMIZER     = "adamw"        # ✅ adamw | adam | sgd | rmsprop | adagrad
    LR_ENCODER    = 1e-5           # ✅ bajo: no destruir pesos ImageNet
    LR_DECODER    = 1e-4           # ✅ decoder + head se entrenan desde cero
    WEIGHT_DECAY  = 1e-4           # ✅
    SGD_MOMENTUM  = 0.9            # ✅ solo si OPTIMIZER == "sgd"
    ADAM_BETAS    = (0.9, 0.999)   # 🔧 actualmente hardcoded a defaults de torch

    # — Scheduler —
    SCHEDULER     = "cosine_warmup"  # ✅ "cosine_warmup" | "poly" | "step" | "constant"
    POLY_POWER    = 0.9              # ✅ exponente del scheduler "poly" (estándar en seg)
    STEP_SIZE     = 30               # ✅ epochs entre decays del scheduler "step"
    STEP_GAMMA    = 0.1              # ✅ factor multiplicativo en cada decay

    # — Estabilidad / regularización —
    GRAD_CLIP_NORM   = 1.0          # ✅ recorte de gradiente
    GRAD_ACCUM_STEPS = 1            # 🔧 acumulación de gradientes (1 = sin acumular)
    LABEL_SMOOTHING  = 0.0          # ✅ suaviza targets one-hot (0.0–0.1 típico). Aplica a
                                    #    CE / OHEM-CE / Weighted-CE / Focal. Dice no lo usa
                                    #    (opera sobre probs, no sobre one-hot).

    # — Promedios de pesos —
    USE_EMA          = False        # ✅ Exponential Moving Average de los pesos
    EMA_DECAY        = 0.9999       # ✅ decay del EMA
    USE_SWA          = False        # 🔧 Stochastic Weight Averaging
    SWA_START_EPOCH  = 40           # 🔧 epoch desde el que SWA empieza a promediar
    SWA_LR           = 5e-5         # 🔧 LR fijo durante SWA

    # — Mixup a nivel de batch (distinto del AUG_MIXUP_ALPHA por imagen) —
    MIXUP_BATCH_ALPHA = 0.0         # 🔧 0.0 = off; típico 0.2–0.4

    # ═══════════════════════════════════════════════════════════════════════
    # 5. PÉRDIDA COMBINADA  (todos editables vía CLI)
    # ═══════════════════════════════════════════════════════════════════════
    # Suma ponderada de las losses con peso > 0.
    # CLI: --ce-weight, --dice-weight, --focal-weight, --lovasz-weight,
    #      --ohem-ce-weight, --weighted-ce-weight.
    # Config GANADORA de la comparativa final (DeepLabV3+ resnet152, COCO):
    # ce0.33 + focal0.33 + lovasz0.33 → mejor mIoU del equipo (0.6039; 0.6283 con TTA).
    CE_WEIGHT          = 0.33      # ✅
    DICE_WEIGHT        = 0.0       # ✅
    FOCAL_WEIGHT       = 0.33      # ✅
    LOVASZ_WEIGHT      = 0.33      # ✅
    OHEM_CE_WEIGHT     = 0.0       # ✅
    WEIGHTED_CE_WEIGHT = 0.0       # ✅

    # — Hiperparámetros de cada loss (CLI: --focal-gamma, --ohem-top-k, --class-weights) —
    FOCAL_GAMMA   = 2.0            # ✅ exponente Focal
    OHEM_TOP_K    = 0.25           # ✅ fracción de píxeles "duros"
    CLASS_WEIGHTS = None           # ✅ None | "auto" | list[float] de NUM_CLASSES
    DICE_SMOOTH   = 1e-6           # 🔧 hardcoded en DiceLoss

    # — Losses futuras —
    BOUNDARY_WEIGHT = 0.0          # 🔧 Boundary Loss (mejora bordes)
    TVERSKY_WEIGHT  = 0.0          # 🔧 Tversky Loss
    TVERSKY_ALPHA   = 0.7          # 🔧 peso falsos positivos
    TVERSKY_BETA    = 0.3          # 🔧 peso falsos negativos

    # ═══════════════════════════════════════════════════════════════════════
    # 6. SAMPLING (DataLoader)
    # ═══════════════════════════════════════════════════════════════════════
    SAMPLER             = "random"  # 🔧 "random" (actual) | "class_balanced" | "hard_example"
    REPEATED_AUG_TIMES  = 1         # 🔧 repeated augmentation: 1 = una augmentation por sample

    # ═══════════════════════════════════════════════════════════════════════
    # 7. VALIDACIÓN / EVALUACIÓN
    # ═══════════════════════════════════════════════════════════════════════
    # Métricas extra (mIoU + IoU por clase ya están activos).
    LOG_PIXEL_ACCURACY  = False    # ✅ accuracy por píxel
    LOG_F1_PER_CLASS    = False    # ✅ F1 / Dice score por clase
    LOG_BOUNDARY_IOU    = False    # ✅ IoU restringido a píxeles de borde

    # — Test-Time Augmentation (en evaluate.py; el flag --tta también lo activa) —
    USE_TTA             = False    # ✅ activar TTA en evaluate.py
    TTA_HFLIP           = True     # ✅ promediar logits con hflip
    TTA_SCALES          = (0.75, 1.0, 1.25)  # ✅ multi-escala

    # — Inferencia con sliding window (para imágenes grandes) —
    USE_SLIDING_WINDOW       = False  # 🔧
    SLIDING_WINDOW_OVERLAP   = 0.25   # 🔧 fracción de solapamiento entre tiles

    # — CRF post-processing (clásico, mejora bordes) —
    USE_CRF_POSTPROCESS = False    # 🔧 requiere pydensecrf

    # ═══════════════════════════════════════════════════════════════════════
    # 8. OPTIMIZACIONES DE GPU (solo efecto en CUDA; todo opt-out)
    # ═══════════════════════════════════════════════════════════════════════
    USE_AMP         = True         # ✅ mixed precision fp16
    COMPILE         = True         # ✅ torch.compile
    CHANNELS_LAST   = True         # ✅ memory format channels_last
    CUDNN_BENCHMARK = True         # ✅ cuDNN autotune (input fijo)

    # ═══════════════════════════════════════════════════════════════════════
    # 9. LOGGING / WANDB
    # ═══════════════════════════════════════════════════════════════════════
    WANDB_PROJECT       = "finetuning"  # ✅ proyecto de W&B donde aparecen los runs
    WANDB_LOG_GRADIENTS = "all"         # ✅ "all" | "gradients" | "parameters" | None
    WANDB_LOG_FREQ      = 50            # ✅ frecuencia (en steps) del watch

    LOG_PREDICTION_SAMPLES = False   # 🔧 sube N imágenes (input/GT/pred) por epoch a W&B
    LOG_N_PRED_SAMPLES     = 4       # 🔧 cuántas
    LOG_CONFUSION_MATRIX   = False   # 🔧 sube confusion matrix por epoch a W&B
    LOG_GRAD_NORMS         = False   # 🔧 norma de gradiente por param group

    # ═══════════════════════════════════════════════════════════════════════
    # 10. CHECKPOINTING
    # ═══════════════════════════════════════════════════════════════════════
    CKPT_DIR              = "checkpoints"  # ✅ carpeta donde se guardan los checkpoints
    SAVE_BEST_ONLY        = True           # ✅ siempre se guarda el de mejor mIoU
    SAVE_EVERY_N_EPOCHS   = 0              # ✅ 0 = nunca; N = cada N epochs snapshot
    RESUME_FROM           = None           # 🔧 ruta de checkpoint para resume

    # ═══════════════════════════════════════════════════════════════════════
    # 11. REPRODUCIBILIDAD
    # ═══════════════════════════════════════════════════════════════════════
    SEED          = 42             # ✅
    DETERMINISTIC = False          # 🔧 torch.use_deterministic_algorithms (ralentiza ~10–20%)
