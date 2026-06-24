from .deeplab import (ASPP, ConcatFusion, CrossAttentionFusion, DeepLabV3Plus,
                      DilatedEncoder)
from .unet import DecoderBlock, Encoder, UNet


def build_model(num_classes: int, cfg) -> "torch.nn.Module":
    """Construye el modelo según `cfg.DECODER_TYPE` ("unet" | "deeplabv3plus").

    Lee del cfg los hiperparámetros relevantes a cada arquitectura. Centraliza
    la decisión en un solo sitio para que `main.py` y `evaluate.py` no diverjan.
    """
    dec = str(getattr(cfg, "DECODER_TYPE", "unet")).lower()
    if dec == "unet":
        return UNet(
            num_classes     = num_classes,
            backbone        = cfg.BACKBONE,
            pretrained      = cfg.PRETRAINED,
            decoder_dropout = getattr(cfg, "DECODER_DROPOUT", 0.0),
        )
    if dec in ("deeplabv3plus", "deeplab", "deeplabv3+"):
        return DeepLabV3Plus(
            num_classes       = num_classes,
            backbone          = cfg.BACKBONE,
            pretrained        = cfg.PRETRAINED,
            output_stride     = getattr(cfg, "DEEPLAB_OUTPUT_STRIDE", 16),
            aspp_out_channels = getattr(cfg, "DEEPLAB_ASPP_OUT", 256),
            decoder_low_ch    = getattr(cfg, "DEEPLAB_DECODER_LOW_CH", 48),
            decoder_dropout   = getattr(cfg, "DECODER_DROPOUT", 0.0),
            aspp_dropout      = getattr(cfg, "DEEPLAB_ASPP_DROPOUT", 0.1),
            use_attention     = getattr(cfg, "USE_ATTENTION", False),
        )
    raise ValueError(f"DECODER_TYPE = {dec!r} no soportado. "
                     "Usa 'unet' o 'deeplabv3plus'.")


__all__ = [
    "UNet", "Encoder", "DecoderBlock",
    "DeepLabV3Plus", "DilatedEncoder", "ASPP",
    "ConcatFusion", "CrossAttentionFusion",
    "build_model",
]
