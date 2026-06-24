import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Canals de cada nivell per a cada backbone.
# Format: (bottleneck, skip3, skip2, skip1, skip0)
# ResNet18/34 usen BasicBlock (canals petits); ResNet50+ usen Bottleneck (canals grans)
_BACKBONE_CHANNELS = {
    "resnet18":  (512,  256, 128, 64,  64),
    "resnet34":  (512,  256, 128, 64,  64),
    "resnet50":  (2048, 1024, 512, 256, 64),
    "resnet101": (2048, 1024, 512, 256, 64),
    "resnet152": (2048, 1024, 512, 256, 64),
}

_BACKBONE_BUILDERS = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2),
    "resnet152": (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V2),
}


class Encoder(nn.Module):
    """
    EXPLICACIÓ SIMPLE: L'Encoder és la primera meitat de U-Net que comprimeix la imatge.
    Usa un backbone preentrenat en ImageNet. Extreu característiques en 5 nivells.
    Suporta: resnet18 | resnet34 | resnet50 | resnet101 | resnet152
    Retorna el bottleneck i les característiques intermèdies per a skip connections.
    """
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone not in _BACKBONE_BUILDERS:
            raise ValueError(
                f"Backbone '{backbone}' no soportado. "
                f"Opciones: {list(_BACKBONE_BUILDERS.keys())}"
            )
        builder, weights_enum = _BACKBONE_BUILDERS[backbone]
        weights = weights_enum if pretrained else None
        net = builder(weights=weights)

        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.channels = _BACKBONE_CHANNELS[backbone]

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4, [x3, x2, x1, x0]


class DecoderBlock(nn.Module):
    """
    EXPLICACIÓ SIMPLE: Un bloc del Decoder que restaura la resolució.
    1. Fa la imatge més gran (upsampling) amb ConvTranspose2d
    2. Combina amb característiques de l'Encoder (skip connection)
    3. Aplica convolucions per refinar els detalls
    4. (Opcional) Dropout2d al final per regularitzar el decoder
    """
    def __init__(self, in_ch, skip_ch, out_ch, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        layers = [
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    EXPLICACIÓ SIMPLE: La xarxa U-Net completa per a segmentació semàntica.
    Combina Encoder + Decoder amb skip connections.
    El backbone es configura via el paràmetre 'backbone':
        resnet18 | resnet34 | resnet50 | resnet101 | resnet152
    Entrada: imatge RGB (B, 3, H, W)
    Sortida: logits per píxel (B, num_classes, H, W)
    """
    def __init__(self, num_classes, backbone="resnet50", pretrained=True,
                 decoder_dropout: float = 0.0):
        super().__init__()
        self.encoder = Encoder(backbone=backbone, pretrained=pretrained)
        b, s3, s2, s1, s0 = self.encoder.channels

        self.dec1 = DecoderBlock(b,    s3, b // 4,  dropout=decoder_dropout)
        self.dec2 = DecoderBlock(b//4, s2, b // 8,  dropout=decoder_dropout)
        self.dec3 = DecoderBlock(b//8, s1, b // 16, dropout=decoder_dropout)
        self.dec4 = DecoderBlock(b//16, s0, b // 32, dropout=decoder_dropout)
        self.head = nn.Conv2d(b // 32, num_classes, kernel_size=1)

    def forward(self, x):
        in_size = x.shape[-2:]
        bottleneck, skips = self.encoder(x)   # skips = [x3, x2, x1, x0]
        d = self.dec1(bottleneck, skips[0])
        d = self.dec2(d,          skips[1])
        d = self.dec3(d,          skips[2])
        d = self.dec4(d,          skips[3])
        out = self.head(d)
        return F.interpolate(out, size=in_size, mode="bilinear", align_corners=False)
