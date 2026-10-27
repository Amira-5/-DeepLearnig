import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNNEncoder(nn.Module):
    """CNN encoder trained from scratch (the baseline)."""

    def __init__(self, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        blocks = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(conv_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.out_channels = in_ch

    def forward(self, x):
        return self.features(x)