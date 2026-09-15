import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

from utils.config import CONFIG, MODEL_CFG, NUM_CLASSES, setup_logging

logger = setup_logging("model")


def build_model(
    num_classes: int = None,
    architecture: str = None,
    pretrained: bool = None,
    dropout: float = None,
) -> nn.Module:
    if num_classes is None:
        num_classes = NUM_CLASSES
    if architecture is None:
        architecture = MODEL_CFG["architecture"]
    if pretrained is None:
        pretrained = MODEL_CFG["pretrained"]
    if dropout is None:
        dropout = MODEL_CFG["dropout"]

    if architecture == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        logger.info(f"Built MobileNetV2 (pretrained={pretrained}, dropout={dropout})")

    elif architecture == "mobilenetv3small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # Replace only the final linear layer, preserving the built-in
        # pooling / feature-reduction head (matches train.py's architecture).
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        logger.info(
            f"Built MobileNetV3-Small (pretrained={pretrained}, dropout={dropout})"
        )

    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        logger.info(f"Built ResNet18 (pretrained={pretrained}, dropout={dropout})")

    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        logger.info(f"Built EfficientNet-B0 (pretrained={pretrained}, dropout={dropout})")

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_m": f"{total / 1e6:.2f}M",
        "trainable_m": f"{trainable / 1e6:.2f}M",
    }


def freeze_backbone(model: nn.Module, freeze_until: str = None):
    for name, param in model.named_parameters():
        param.requires_grad = False

    if freeze_until:
        unfreeze = False
        for name, param in model.named_parameters():
            if freeze_until in name:
                unfreeze = True
            if unfreeze:
                param.requires_grad = True
    else:
        last_module = list(model.children())[-1]
        for param in last_module.parameters():
            param.requires_grad = True

    info = count_parameters(model)
    logger.info(
        f"Frozen backbone: {info['frozen']} params frozen, "
        f"{info['trainable']} params trainable"
    )


if __name__ == "__main__":
    model = build_model()
    info = count_parameters(model)
    print(f"Model: {MODEL_CFG['architecture']}")
    print(f"Parameters: {info}")

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Predictions:  {out.argmax(dim=1).tolist()}")
