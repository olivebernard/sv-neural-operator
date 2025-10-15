from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.special import ndtr
from torch import Tensor, nn

from .layers import FNOBlock

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HestonParameters:
    spot: float = 100.0
    rate: float = 0.02
    v0: float = 0.045
    kappa: float = 1.7
    theta: float = 0.04
    vol_of_vol: float = 0.48
    rho: float = -0.68


def chebyshev_strike_grid(n: int = 48, low: float = 60.0, high: float = 140.0) -> FloatArray:
    if n < 8 or not high > low:
        raise ValueError("Grille de strike invalide.")
    theta = np.linspace(np.pi, 0.0, n)
    nodes = 0.5 * (low + high) + 0.5 * (high - low) * np.cos(theta)
    return np.sort(nodes.astype(np.float64))


def _effective_volatility(strikes: FloatArray, maturity: float, params: HestonParameters) -> FloatArray:
    log_moneyness = np.log(np.maximum(strikes, 1e-8) / params.spot)
    base = np.sqrt(params.theta + (params.v0 - params.theta) * np.exp(-params.kappa * maturity))
    skew = 1.0 - 0.34 * params.rho * log_moneyness
    curvature = 1.0 + 0.22 * params.vol_of_vol * log_moneyness**2 / np.sqrt(maturity + 0.08)
    return np.maximum(0.06, base * skew * curvature)


def reference_surface(
    strikes: FloatArray,
    maturities: FloatArray,
    params: HestonParameters | None = None,
) -> FloatArray:
    p = params or HestonParameters()
    k = np.asarray(strikes, dtype=np.float64)
    t = np.asarray(maturities, dtype=np.float64)
    rows = []
    for maturity in t:
        sigma = _effective_volatility(k, float(maturity), p)
        sqrt_t = np.sqrt(maturity)
        d1 = (np.log(p.spot / k) + (p.rate + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        rows.append(p.spot * ndtr(d1) - k * np.exp(-p.rate * maturity) * ndtr(d2))
    return np.asarray(rows, dtype=np.float64)


class PricingFNO(nn.Module):
    def __init__(self, input_channels: int = 4, width: int = 24, depth: int = 4) -> None:
        super().__init__()
        self.lift = nn.Conv2d(input_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes_x=12, modes_y=8) for _ in range(depth)])
        self.project = nn.Sequential(
            nn.Conv2d(width, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, fields: Tensor) -> Tensor:
        x = self.lift(fields)
        for block in self.blocks:
            x = block(x)
        return self.project(x).squeeze(1)


def build_parameter_field(
    strikes: FloatArray,
    maturities: FloatArray,
    params: HestonParameters | None = None,
) -> Tensor:
    p = params or HestonParameters()
    k, t = np.meshgrid(strikes / p.spot, maturities, indexing="xy")
    fields = np.stack(
        [k, t, np.full_like(k, p.v0), np.full_like(k, p.rho)], axis=0
    )
    return torch.as_tensor(fields[None, ...], dtype=torch.float32)
