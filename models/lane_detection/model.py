"""
DeepLabV3+ with ASPP — fixed for batch_size=1 and eval/sanity-check usage.

Fixes applied
─────────────
1. gap branch uses GroupNorm instead of BatchNorm2d (BN can't handle 1×1 spatial maps).
2. build_and_check() calls model.eval() before the single-image sanity-check.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    IMG_HEIGHT   = 512
    IMG_WIDTH    = 512
    NUM_CLASSES  = 21          # VOC; change freely
    ASPP_OUT_CH  = 256
    DECODER_CH   = 48
    DROPOUT      = 0.5

cfg = Config()


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU6.
    Safe for full spatial maps (H/4, H/8, H/16, H/32).

    padding is passed directly to Conv2d — do NOT multiply by dilation here.
    For a 3×3 atrous conv with dilation=d, pass padding=d (same-padding formula:
    padding = dilation * (kernel - 1) // 2 = d*1 = d).
    Multiplying again (padding * dilation) would over-pad and produce a larger
    output than the input, causing the cat() size mismatch in ASPP.
    """
    def __init__(self, in_ch, out_ch, kernel=3, stride=1,
                 padding=1, dilation=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, dilation=dilation, bias=bias),  # ← fixed
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ──────────────────────────────────────────────────────────────────────────────
# ASPP module  ← main fix lives here
# ──────────────────────────────────────────────────────────────────────────────

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    The global-average-pool (gap) branch collapses spatial dims to 1×1 before
    normalising.  BatchNorm2d in training mode requires >1 spatial value, so it
    crashes with any batch size when the feature map becomes 1×1.

    Fix: replace BatchNorm2d in the gap branch with GroupNorm(num_groups, ch).
    GroupNorm normalises over channel groups independently of spatial size, so
    it works correctly on 1×1 tensors.
    """

    def __init__(self, in_ch: int, out_ch: int = 256,
                 dilations: tuple = (6, 12, 18)):
        super().__init__()

        # 1×1 convolution
        self.conv1 = ConvBnRelu(in_ch, out_ch, kernel=1, padding=0)

        # Atrous convolutions at three scales
        self.atrous = nn.ModuleList([
            ConvBnRelu(in_ch, out_ch, dilation=d, padding=d) for d in dilations
        ])

        # ── Global Average Pool branch ──────────────────────────────────────
        # BN after AdaptiveAvgPool2d(1) → always 1×1 → BN training crash.
        # GroupNorm does NOT depend on spatial resolution, so it is safe here.
        num_groups = 32  # must divide out_ch evenly (256 / 32 = 8 ✓)
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(num_groups, out_ch),   # ← was nn.BatchNorm2d(out_ch)
            nn.ReLU6(inplace=True),
        )
        # ────────────────────────────────────────────────────────────────────

        # Project concatenated branches back to out_ch
        total_branches = 1 + len(dilations) + 1   # conv1 + atrous + gap
        self.project = nn.Sequential(
            ConvBnRelu(out_ch * total_branches, out_ch, kernel=1, padding=0),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        gap = F.interpolate(self.gap(x), size=(h, w),
                            mode="bilinear", align_corners=False)
        feats = [self.conv1(x)] + [a(x) for a in self.atrous] + [gap]
        return self.project(torch.cat(feats, dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# Encoder (ResNet-50 backbone)
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """ResNet-50 with dilated conv in layer3/layer4 (output_stride=16)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        # Stem + layer1 → low-level features at stride 4
        self.layer0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # stride 4,  256 ch  (low-level)
        self.layer2 = backbone.layer2   # stride 8,  512 ch
        self.layer3 = backbone.layer3   # stride 16, 1024 ch  (dilated)
        self.layer4 = backbone.layer4   # stride 16, 2048 ch  (dilated)

        # Apply dilation to keep output_stride=16
        self._apply_dilation(self.layer3, dilation=2, stride=1)
        self._apply_dilation(self.layer4, dilation=4, stride=1)

    @staticmethod
    def _apply_dilation(layer, dilation, stride):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding  = (dilation, dilation)
                    m.stride   = (stride, stride)
            elif isinstance(m, nn.BatchNorm2d):
                pass
        # Fix the first downsample (stride conv) so spatial size stays at /16
        first_block = layer[0]
        if first_block.downsample is not None:
            first_block.downsample[0].stride = (stride, stride)
        first_block.conv2.stride = (stride, stride)

    def forward(self, x):
        x = self.layer0(x)
        low = self.layer1(x)    # /4  — returned for decoder skip connection
        x   = self.layer2(low)
        x   = self.layer3(x)
        x   = self.layer4(x)   # /16 — goes into ASPP
        return x, low


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, low_ch: int = 256, aspp_ch: int = 256,
                 num_classes: int = 21, decoder_ch: int = 48):
        super().__init__()
        self.low_proj = ConvBnRelu(low_ch, decoder_ch, kernel=1, padding=0)

        self.refine = nn.Sequential(
            ConvBnRelu(aspp_ch + decoder_ch, 256),
            ConvBnRelu(256, 256),
        )
        self.cls = nn.Conv2d(256, num_classes, 1)

    def forward(self, aspp_feat, low_feat):
        low = self.low_proj(low_feat)                           # /4
        h, w = low.shape[-2:]
        aspp_up = F.interpolate(aspp_feat, size=(h, w),
                                mode="bilinear", align_corners=False)
        x = torch.cat([aspp_up, low], dim=1)
        x = self.refine(x)
        return self.cls(x)                                      # /4 logits


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = cfg.NUM_CLASSES,
                 pretrained: bool = True):
        super().__init__()
        self.encoder = Encoder(pretrained=pretrained)
        self.aspp    = ASPP(in_ch=2048, out_ch=cfg.ASPP_OUT_CH)
        self.decoder = Decoder(
            low_ch=256,
            aspp_ch=cfg.ASPP_OUT_CH,
            num_classes=num_classes,
            decoder_ch=cfg.DECODER_CH,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        enc, low = self.encoder(x)
        aspp     = self.aspp(enc)
        logits   = self.decoder(aspp, low)
        # Upsample back to input resolution
        return F.interpolate(logits, size=(h, w),
                             mode="bilinear", align_corners=False)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = cfg.NUM_CLASSES,
                pretrained: bool = True) -> DeepLabV3Plus:
    return DeepLabV3Plus(num_classes=num_classes, pretrained=pretrained)


def build_and_check():
    """Sanity-check: single-image forward pass (eval mode)."""
    model = build_model(pretrained=False)

    # ── Fix 2: always call eval() before a single-image sanity check ────────
    model.eval()
    # ────────────────────────────────────────────────────────────────────────

    x = torch.randn(1, 3, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    with torch.no_grad():
        y = model(x)

    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(y.shape)}")
    assert y.shape == (1, cfg.NUM_CLASSES, cfg.IMG_HEIGHT, cfg.IMG_WIDTH), \
        f"Unexpected output shape: {y.shape}"
    print("✓ Sanity check passed.")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4):
    """Separate LRs: 10× lower for the pretrained backbone."""
    backbone_ids = {id(p) for p in model.encoder.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) in backbone_ids]
    head_params     = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ], weight_decay=weight_decay)


def get_criterion(ignore_index: int = 255):
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_and_check()