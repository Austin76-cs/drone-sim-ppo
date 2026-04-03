"""Lightweight U-Net for drone gate heatmap prediction.

Input:  RGB image  (B, 3, H, W)  — values in [0, 1]
Output: heatmap    (B, 1, H, W)  — values in [0, 1] (sigmoid)

Architecture: 4-level encoder/decoder with skip connections.
Designed for 160×120 input on a 3080 Ti (fits easily in VRAM).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims don't match (can happen with odd input sizes)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class GateUNet(nn.Module):
    """U-Net that predicts a gate-center heatmap from a drone camera image."""

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        b = base_ch
        self.inc   = _DoubleConv(3, b)          # (B, b,   H,   W)
        self.down1 = _Down(b,     b * 2)         # (B, 2b,  H/2, W/2)
        self.down2 = _Down(b * 2, b * 4)         # (B, 4b,  H/4, W/4)
        self.down3 = _Down(b * 4, b * 8)         # (B, 8b,  H/8, W/8)
        self.down4 = _Down(b * 8, b * 16)        # (B, 16b, H/16, W/16)

        self.up1 = _Up(b * 16, b * 8,  b * 8)   # (B, 8b,  H/8, W/8)
        self.up2 = _Up(b * 8,  b * 4,  b * 4)   # (B, 4b,  H/4, W/4)
        self.up3 = _Up(b * 4,  b * 2,  b * 2)   # (B, 2b,  H/2, W/2)
        self.up4 = _Up(b * 2,  b,      b)        # (B, b,   H,   W)

        self.head = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return torch.sigmoid(self.head(x))

    @staticmethod
    def from_checkpoint(path: str, device: str = "cuda") -> "GateUNet":
        model = GateUNet()
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model"])
        return model.to(device).eval()
