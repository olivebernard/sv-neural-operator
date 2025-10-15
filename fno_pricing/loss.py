from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    data: Tensor
    monotonicity: Tensor
    convexity: Tensor


class ArbitrageFreeLoss(nn.Module):
    """Erreur de surface avec contraintes discrètes de forme."""

    def __init__(
        self,
        lambda_monotonicity: float = 10.0,
        lambda_convexity: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_mono = float(lambda_monotonicity)
        self.lambda_convexity = float(lambda_convexity)

    def breakdown(
        self,
        pred_prices: Tensor,
        target_prices: Tensor,
        strikes: Tensor | None = None,
    ) -> LossBreakdown:
        data = F.mse_loss(pred_prices, target_prices)
        grad_k = pred_prices[..., 2:] - pred_prices[..., :-2]
        monotonicity = torch.relu(grad_k).mean()
        curvature = pred_prices[..., 2:] - 2.0 * pred_prices[..., 1:-1] + pred_prices[..., :-2]
        convexity = torch.relu(-curvature).mean()
        total = data + self.lambda_mono * monotonicity + self.lambda_convexity * convexity
        return LossBreakdown(total=total, data=data, monotonicity=monotonicity, convexity=convexity)

    def forward(
        self,
        pred_prices: Tensor,
        target_prices: Tensor,
        strikes: Tensor | None = None,
    ) -> Tensor:
        return self.breakdown(pred_prices, target_prices, strikes).total
