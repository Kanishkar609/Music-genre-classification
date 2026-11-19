"""Six CNN architectures for mel-spectrogram genre classification.

Every model takes a (batch, 1, N_MELS, SEG_FRAMES) tensor and returns logits.
They intentionally differ in inductive bias / capacity so the ensemble has
something to gain from.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(cin: int, cout: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(w), inplace=True))).view(b, c, 1, 1)
        return x * w


class BaselineCNN(nn.Module):
    """Compact 3-block VGG-lite. Fast and strong on mel patches."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2),
            conv_bn(32, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.35), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class DeepCNN(nn.Module):
    """Four blocks with widening channels - more capacity, still CPU friendly."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 32), nn.MaxPool2d(2),
            conv_bn(32, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), nn.MaxPool2d(2),
            conv_bn(128, 256), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class VGGStyleCNN(nn.Module):
    """Double 3x3 convolutions per block (classic VGG recipe, scaled down)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 32), conv_bn(32, 32), nn.MaxPool2d(2),
            conv_bn(32, 64), conv_bn(64, 64), nn.MaxPool2d(2),
            conv_bn(64, 128), conv_bn(128, 128), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv_bn(cin, cout, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        res = x if self.skip is None else self.skip(x)
        return F.relu(out + res, inplace=True)


class ResNetSmall(nn.Module):
    """Small residual network - residual connections help with spectrogram noise."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = conv_bn(1, 32)
        self.blocks = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class DepthwiseSeparable(nn.Module):
    """MobileNet-style separable convolutions: fewest FLOPs per epoch."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            self._sep_block(1, 32, stride=2),
            self._sep_block(32, 64, stride=2),
            self._sep_block(64, 128, stride=2),
            self._sep_block(128, 256, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.35), nn.Linear(256, num_classes),
        )

    @staticmethod
    def _sep_block(cin: int, cout: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.features(x))


class AttentionCNN(nn.Module):
    """SE-attention backbone + learned attention pooling over time/frequency."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(1, 32), SEBlock(32), nn.MaxPool2d(2),
            conv_bn(32, 64), SEBlock(64), nn.MaxPool2d(2),
            conv_bn(64, 128), SEBlock(128), nn.MaxPool2d(2),
        )
        self.attn = nn.Conv2d(128, 1, 1)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        f = self.features(x)                       # B,128,H,W
        w = torch.softmax(self.attn(f).flatten(2), dim=-1)   # B,1,HW
        pooled = torch.bmm(f.flatten(2), w.transpose(1, 2)).squeeze(-1)  # B,128
        return self.fc(self.drop(pooled))


MODELS: dict[str, type[nn.Module]] = {
    "baseline": BaselineCNN,
    "deep": DeepCNN,
    "vgg": VGGStyleCNN,
    "resnet": ResNetSmall,
    "separable": DepthwiseSeparable,
    "attention": AttentionCNN,
}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown CNN '{name}'. Options: {list(MODELS)}")
    return MODELS[name](num_classes=num_classes)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
