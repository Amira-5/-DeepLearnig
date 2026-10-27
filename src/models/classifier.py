import torch.nn as nn


class ImageClassifier(nn.Module):
    """Any encoder + a classification head."""

    def __init__(self, encoder, num_classes=200, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(encoder.out_channels, num_classes)

    def forward(self, x):
        x = self.encoder(x)               # [B, C, H, W] feature map
        x = self.pool(x).flatten(1)       # [B, C]
        return self.fc(self.dropout(x))   # [B, num_classes]