from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionPreservingTokenization(nn.Module):
    VALID = {
        "none", "s2d", "local", "full",
        "dw_pw", "amp", "dir", "add", "mul",
    }

    def __init__(
        self,
        in_channels=3,
        hidden_channels=24,
        unshuffle_factor=2,
        variant="full",
        residual_scale=0.5,
    ):
        super().__init__()
        if variant not in self.VALID:
            raise ValueError(f"Unsupported MPT/PMME variant: {variant}")

        self.variant = variant
        self.r = int(unshuffle_factor)
        self.residual_scale = float(residual_scale)
        c = in_channels * self.r * self.r

        self.s2d = nn.PixelUnshuffle(self.r)
        self.d2s = nn.PixelShuffle(self.r)

        self.in_proj = nn.Sequential(
            nn.Conv2d(c, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.local_filter = nn.Sequential(
            nn.Conv2d(
                hidden_channels, hidden_channels, 3,
                padding=1, groups=hidden_channels, bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        gate_hidden = max(hidden_channels // 4, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, gate_hidden, 1),
            nn.GELU(),
            nn.Conv2d(gate_hidden, hidden_channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    @staticmethod
    def _motion_cues(x):
        eps = 1e-6
        u, v = x[:, 0:1], x[:, 1:2]
        mag = torch.sqrt(u.square() + v.square() + eps)

        mean = mag.mean((2, 3), keepdim=True)
        std = mag.std(
            (2, 3), keepdim=True, unbiased=False
        ).clamp_min(eps)
        amplitude = torch.sigmoid((mag - mean) / std)

        dx, dy = u / mag, v / mag
        ldx = F.avg_pool2d(dx, 3, 1, 1)
        ldy = F.avg_pool2d(dy, 3, 1, 1)
        lnorm = torch.sqrt(ldx.square() + ldy.square() + eps)

        cosine = (dx * ldx + dy * ldy) / lnorm
        direction = ((cosine + 1.0) * 0.5).clamp(0, 1)
        direction = direction * (mag > 1e-5).to(x.dtype)

        return amplitude, direction

    def forward(self, x):
        if self.variant == "none":
            return x

        source = x
        if self.variant in {"amp", "dir", "add", "mul"}:
            amp, direction = self._motion_cues(x)

            if self.variant == "amp":
                response = amp
            elif self.variant == "dir":
                response = direction
            elif self.variant == "add":
                response = 0.5 * (amp + direction)
            else:
                response = amp * direction

            source = x * (0.5 + response)

        z = self.in_proj(self.s2d(source))

        if self.variant in {
            "local", "full", "dw_pw",
            "amp", "dir", "add", "mul",
        }:
            z = self.local_filter(z)

        if self.variant == "full":
            z = z * self.channel_gate(z) * self.spatial_gate(z)

        z = self.d2s(self.out_proj(z))
        return x + self.residual_scale * z
