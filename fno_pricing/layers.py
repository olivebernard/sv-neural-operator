from __future__ import annotations

import torch
from torch import Tensor, nn


class SpectralConv2d(nn.Module):
    """Convolution spectrale tronquée sur les deux axes de la surface."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1.0 / max(in_channels * out_channels, 1)
        shape = (in_channels, out_channels, modes_x, modes_y)
        self.weight_positive = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weight_negative = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _contract(field: Tensor, weight: Tensor) -> Tensor:
        return torch.einsum("bixy,ioxy->boxy", field, weight)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, nx, ny = x.shape
        transformed = torch.fft.rfft2(x)
        out = torch.zeros(
            batch,
            self.out_channels,
            nx,
            ny // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        mx = min(self.modes_x, nx)
        my = min(self.modes_y, ny // 2 + 1)
        out[:, :, :mx, :my] = self._contract(
            transformed[:, :, :mx, :my], self.weight_positive[:, :, :mx, :my]
        )
        out[:, :, -mx:, :my] = self._contract(
            transformed[:, :, -mx:, :my], self.weight_negative[:, :, :mx, :my]
        )
        return torch.fft.irfft2(out, s=(nx, ny))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.local = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)
        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.spectral(x) + self.local(x)))
