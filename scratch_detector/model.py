"""CenterNet-style single-class detector: ResNet18 backbone + upsampling head.

Output stride 4. Three heads over the shared feature map:
  heatmap (1ch)  — object-center probability (logits; sigmoid at decode/loss)
  wh      (2ch)  — box width/height in output-grid units
  offset  (2ch)  — sub-cell center offset in [0, 1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

OUTPUT_STRIDE = 4


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _head(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_ch, 1),
    )


class CenterNetLite(nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        backbone = resnet18(weights="DEFAULT" if pretrained_backbone else None)
        # conv1..layer4: stride 32, 512 channels
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # stride 32 → 4
        self.up = nn.Sequential(UpBlock(512, 256), UpBlock(256, 128), UpBlock(128, 64))
        self.heatmap = _head(64, 1)
        self.wh = _head(64, 2)
        self.offset = _head(64, 2)
        # Bias init so the heatmap starts near-empty (focal-loss stability)
        self.heatmap[-1].bias.data.fill_(-4.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.up(self.backbone(x))
        return {"heatmap": self.heatmap(feat), "wh": self.wh(feat), "offset": self.offset(feat)}


@torch.no_grad()
def decode(
    outputs: dict[str, torch.Tensor],
    top_k: int = 64,
    conf_threshold: float = 0.30,
) -> list[torch.Tensor]:
    """Decode a batch → per-image (n, 5) tensors: x1, y1, x2, y2, score (input px).

    Max-pool NMS: a cell survives only if it is the local maximum in its
    3x3 neighbourhood — this replaces IoU-based NMS in CenterNet.
    """
    heat = torch.sigmoid(outputs["heatmap"])
    local_max = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
    heat = heat * (local_max == heat)

    batch, _, height, width = heat.shape
    results = []
    for b in range(batch):
        scores, indices = heat[b, 0].flatten().topk(min(top_k, height * width))
        keep = scores >= conf_threshold
        scores, indices = scores[keep], indices[keep]
        ys = (indices // width).float()
        xs = (indices % width).float()
        wh = outputs["wh"][b, :, ys.long(), xs.long()]  # (2, n)
        off = outputs["offset"][b, :, ys.long(), xs.long()]  # (2, n)
        cx = (xs + off[0].clamp(0, 1)) * OUTPUT_STRIDE
        cy = (ys + off[1].clamp(0, 1)) * OUTPUT_STRIDE
        w = wh[0].clamp(min=0) * OUTPUT_STRIDE
        h = wh[1].clamp(min=0) * OUTPUT_STRIDE
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, scores], dim=1)
        results.append(boxes)
    return results
