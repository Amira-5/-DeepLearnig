import torch
import torch.nn as nn
import torchvision.models as models


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNNEncoder(nn.Module):
    """CNN encoder trained from scratch (the baseline)[cite: 1]."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(conv_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class TransferLearningEncoder(nn.Module):
    """Pretrained backbone encoder for transfer learning[cite: 5, 6]."""

    def __init__(self, name: str = "resnet18", pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.name = name.lower()

        if self.name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            # Remove global average pooling and FC layer
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.out_channels = 512

        elif self.name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.out_channels = 2048

        elif self.name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            self.features = backbone.features
            self.out_channels = 1280

        else:
            raise ValueError(f"Unsupported transfer learning backbone: {name}")

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)