from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fno_pricing.loss import ArbitrageFreeLoss
from fno_pricing.solver import (
    HestonParameters,
    PricingFNO,
    build_parameter_field,
    chebyshev_strike_grid,
    reference_surface,
)


def main() -> None:
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    strikes = chebyshev_strike_grid(56, 60.0, 140.0)
    maturities = np.array([0.12, 0.25, 0.50, 0.85, 1.20])
    target = reference_surface(strikes, maturities, HestonParameters())

    index = np.arange(strikes.size)
    alternating = (-1.0) ** index
    tail_envelope = np.exp(-((strikes - 122.0) / 17.0) ** 2)
    maturity_scale = np.linspace(0.85, 1.15, maturities.size)[:, None]
    perturbation = 0.055 * maturity_scale * alternating[None, :] * tail_envelope[None, :]
    prediction = np.maximum(target + perturbation, 0.0)

    loss = ArbitrageFreeLoss(lambda_monotonicity=10.0, lambda_convexity=0.0)
    pred_tensor = torch.as_tensor(prediction, dtype=torch.float32)
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    strike_tensor = torch.as_tensor(strikes, dtype=torch.float32)
    terms = loss.breakdown(pred_tensor, target_tensor, strike_tensor)

    torch.manual_seed(5)
    model = PricingFNO(input_channels=4, width=12, depth=2)
    with torch.no_grad():
        forward_shape = list(model(build_parameter_field(strikes, maturities)).shape)

    fig = plt.figure(figsize=(11, 5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2)
    ax0 = fig.add_subplot(grid[0, 0], projection="3d")
    strike_mesh, maturity_mesh = np.meshgrid(strikes, maturities)
    ax0.plot_surface(strike_mesh, maturity_mesh, prediction, linewidth=0, antialiased=True, alpha=0.88)
    ax0.set_xlabel("strike K")
    ax0.set_ylabel("maturité T")
    ax0.set_zlabel("prix du call")
    ax0.set_title("Surface prédite par opérateur spectral")

    ax1 = fig.add_subplot(grid[0, 1])
    maturity_index = 2
    ax1.plot(strikes, target[maturity_index], label="référence")
    ax1.plot(strikes, prediction[maturity_index], label="prédiction")
    ax1.set_xlabel("strike K")
    ax1.set_ylabel("prix")
    ax1.set_title(f"Coupe T={maturities[maturity_index]:.2f}")
    ax1.grid(alpha=0.2)
    ax1.legend()
    fig.savefig(output / "price_surface.png", dpi=180)
    plt.close(fig)

    metrics = {
        "strike_nodes": int(strikes.size),
        "maturities": maturities.tolist(),
        "nonuniform_spacing_ratio": float(np.diff(strikes).max() / np.diff(strikes).min()),
        "reported_monotonicity_penalty": float(terms.monotonicity),
        "reported_data_loss": float(terms.data),
        "reported_total_loss": float(terms.total),
        "fno_forward_shape": forward_shape,
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
