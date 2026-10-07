import torch
import torch.nn as nn
from torchvision import models

def build_resnet18_feature_extractor(num_classes: int = 38, freeze_backbone: bool = True):
    """
    Loads official pretrained ResNet18, freezes the backbone layers,
    and replaces the final classification head (fc) for plant disease classification.
    """
    # Load official ImageNet pretrained ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Freeze backbone parameters
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final linear layer (backbone has 512 output features)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Ensure new classification head parameters are trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_parameters(model: nn.Module):
    """
    Returns total parameters and trainable parameters count.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen
    }
