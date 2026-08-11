import torch
import torch.nn as nn
import torchvision.models as models


class BaselineExtractor(nn.Module):
    """
    Pretrained ResNet50 with FC removed.
    Outputs 2048-dimensional embeddings.
    Used as zero-shot baseline before contrastive learning.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        base = models.resnet50(pretrained=True)

        if in_channels != 3:
            old_conv = base.conv1

            new_conv = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            with torch.no_grad():
                for i in range(in_channels):
                    new_conv.weight[:, i, :, :] = old_conv.weight[:, i, :, :]

            base.conv1 = new_conv

        # Remove ResNet classification layer
        base.fc = nn.Identity()

        self.encoder = base

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":

    # Sentinel-2 RGB test
    s2_model = BaselineExtractor(in_channels=3)

    out = s2_model(
        torch.randn(4, 3, 224, 224)
    )

    assert out.shape == (4, 2048)

    print(f"S2 extractor: {out.shape} OK")

    # SAR test
    sar_model = BaselineExtractor(in_channels=2)

    out = sar_model(
        torch.randn(4, 2, 224, 224)
    )

    assert out.shape == (4, 2048)

    print(f"SAR extractor: {out.shape} OK")