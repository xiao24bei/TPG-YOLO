"""TPG-YOLO final texture-to-P5 branch, manuscript Sections 3.3–3.5.

Reconstructed core implementation; no upstream restoration network is included.
"""
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GlobalPriorAdapter(nn.Module):
    """Equations (5)–(7): [B,256] -> [B,C,H,W].

    The 16x16 layout organizes latent components, not image coordinates.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(256, elementwise_affine=False)
        # Bias is retained here, unlike the two joint-encoder convolutions.
        self.projection = nn.Conv2d(1, channels, kernel_size=1, bias=True)

    def forward(self, z_t: Tensor, reference: Tensor) -> Tensor:
        if z_t.ndim != 2 or z_t.shape != (reference.shape[0], 256):
            raise ValueError("z_t must have shape [B,256] and match the feature batch")
        if not z_t.is_floating_point() or not torch.isfinite(z_t).all():
            raise ValueError("z_t must contain finite floating-point values")
        latent = self.norm(z_t.to(reference)).reshape(-1, 1, 16, 16)
        latent = F.interpolate(
            latent, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.projection(latent)


class JointResidualEncoder(nn.Module):
    """Equations (8)–(9): jointly encode the feature and adapted prior."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
        )

    def forward(self, feature: Tensor, adapted_prior: Tensor) -> Tensor:
        return self.layers(torch.cat((feature, adapted_prior), dim=1))


class TextureP5Fusion(nn.Module):
    """Equation (10): P5' = P5 + gamma * Q; P3 and P4 are identical objects.

    The default YOLO11s P5 has 512 channels. Only one branch is instantiated;
    no mixture selector or texture gate is present in the final model.
    """

    def __init__(self, channels: int = 512, gamma_init: float = 0.1):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.adapter = GlobalPriorAdapter(channels)
        self.encoder = JointResidualEncoder(channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, features: Sequence[Tensor], z_t: Tensor) -> list[Tensor]:
        if len(features) != 3:
            raise ValueError("Expected exactly [P3, P4, P5]")
        p3, p4, p5 = features
        if any(f.ndim != 4 for f in features):
            raise ValueError("P3/P4/P5 must be BCHW tensors")
        if any(f.shape[0] != p5.shape[0] for f in features):
            raise ValueError("P3/P4/P5 batch sizes must match")
        if p5.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} P5 channels")
        prior = self.adapter(z_t, p5)
        residual = self.encoder(p5, prior)
        return [p3, p4, p5 + self.gamma * residual]
