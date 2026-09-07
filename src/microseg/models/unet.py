"""U-Net for nuclei segmentation.

A faithful U-Net (Ronneberger et al., 2015) with the modernisations that are now
standard: padded convolutions so input and output share a resolution,
normalisation layers, and a configurable depth/width.

Why U-Net specifically for this problem:

* Skip connections carry high-resolution detail past the bottleneck.  Nuclear
  boundaries are one or two pixels wide, and an encoder-decoder without skips
  cannot reconstruct them from a 16x-downsampled feature map.
* It is fully convolutional, so a model trained on 256 px tiles runs on a whole
  field of arbitrary size at inference -- as long as the dimensions are padded to
  a multiple of ``2**depth`` (see :func:`~microseg.data.transforms.pad_to_multiple`).
* It works with a few hundred annotated images, which is the regime biomedical
  imaging is always in.

``in_channels`` is configurable rather than fixed at 1 so a multiplexed assay
(DAPI + membrane marker) can be fed as a stack; see the multi-channel notes in
the README.

**The optional distance head.**  With ``distance_head=True`` a second 1x1
convolution reads the same decoder features and regresses the normalised
distance to each nucleus' edge.  It shares the entire encoder-decoder because
the two tasks want the same features -- "where does this object end" is the
question behind both -- and a second decoder would triple the parameter count
for no signal the first one lacks.  It is appended to the class logits rather
than returned separately so ``forward`` keeps returning one tensor and every
existing call site still type-checks; :attr:`UNet.n_classes` says where to cut.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from microseg.config import ModelConfig


def _norm_layer(kind: str, channels: int) -> nn.Module:
    """Normalisation factory.

    ``instance`` norm is worth reaching for here: microscopy batches are small
    (large tiles, limited GPU memory) and BatchNorm's statistics get noisy below
    about 8 samples per batch.
    """
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind in ("none", None):
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}; expected batch|instance|none")


class DoubleConv(nn.Module):
    """(conv 3x3 -> norm -> ReLU) x 2, the U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int, norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=norm == "none"),
            _norm_layer(norm, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=norm == "none"),
            _norm_layer(norm, out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            # Dropout2d drops whole feature maps, which is the right granularity
            # for convolutional features -- dropping individual pixels barely
            # regularises when neighbours are highly correlated.
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool then DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, norm: str, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_ch, out_ch, norm, dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upsample, concatenate the skip connection, then DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str, dropout: float, bilinear: bool):
        super().__init__()
        if bilinear:
            # Bilinear + 1x1 conv avoids the checkerboard artifacts that
            # transposed convolutions produce, at slightly lower capacity.
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, in_ch // 2, 1),
            )
            up_out = in_ch // 2
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            up_out = in_ch // 2
        self.conv = DoubleConv(up_out + skip_ch, out_ch, norm, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Odd input sizes leave a one-pixel mismatch after pooling; pad rather
        # than crop so no border information is thrown away.
        diff_y = skip.shape[-2] - x.shape[-2]
        diff_x = skip.shape[-1] - x.shape[-1]
        if diff_y or diff_x:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """Configurable U-Net.

    Outputs raw logits of shape ``(B, out_channels, H, W)``.  With the default
    ``out_channels=3`` the classes are background / interior / boundary.

    When ``distance_head`` is set the tensor is one channel wider: channels
    ``[:out_channels]`` are class logits for a softmax, and the final channel is
    a distance logit for a sigmoid.  Never softmax the whole tensor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        base_filters: int = 32,
        depth: int = 4,
        norm: str = "batch",
        dropout: float = 0.1,
        bilinear_upsample: bool = False,
        distance_head: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.distance_head = bool(distance_head)

        widths = [base_filters * 2**i for i in range(depth + 1)]
        self.stem = DoubleConv(in_channels, widths[0], norm, 0.0)

        self.downs = nn.ModuleList(
            [
                Down(widths[i], widths[i + 1], norm, dropout if i >= depth - 2 else 0.0)
                for i in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Up(widths[i + 1], widths[i], widths[i], norm, 0.0, bilinear_upsample)
                for i in reversed(range(depth))
            ]
        )
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        self.distance = nn.Conv2d(widths[0], 1, 1) if self.distance_head else None
        self.apply(_init_weights)

    @property
    def size_divisor(self) -> int:
        """Input dimensions must be a multiple of this."""
        return 2**self.depth

    @property
    def n_classes(self) -> int:
        """How many leading channels of the output are class logits."""
        return self.out_channels

    @property
    def has_distance(self) -> bool:
        return self.distance is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for i, up in enumerate(self.ups):
            x = up(x, skips[-(i + 2)])

        logits = self.head(x)
        if self.distance is None:
            return logits
        return torch.cat([logits, self.distance(x)], dim=1)


def _init_weights(module: nn.Module) -> None:
    """He initialisation, matched to the ReLU activations."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model(cfg: ModelConfig) -> UNet:
    """Instantiate the model described by a :class:`ModelConfig`."""
    if cfg.name != "unet":
        raise ValueError(f"unknown model {cfg.name!r}; only 'unet' is implemented")
    return UNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_filters=cfg.base_filters,
        depth=cfg.depth,
        norm=cfg.norm,
        dropout=cfg.dropout,
        bilinear_upsample=cfg.bilinear_upsample,
        distance_head=cfg.distance_head,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
