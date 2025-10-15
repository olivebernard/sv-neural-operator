import numpy as np
import torch

from fno_pricing.loss import ArbitrageFreeLoss
from fno_pricing.solver import PricingFNO, build_parameter_field, chebyshev_strike_grid, reference_surface


def test_grid_is_strictly_increasing_and_nonuniform() -> None:
    strikes = chebyshev_strike_grid(32)
    spacing = np.diff(strikes)
    assert np.all(spacing > 0.0)
    assert spacing.max() > 3.0 * spacing.min()


def test_reference_surface_has_expected_shape() -> None:
    strikes = chebyshev_strike_grid(24)
    maturities = np.array([0.25, 0.5, 1.0])
    surface = reference_surface(strikes, maturities)
    assert surface.shape == (3, 24)
    assert np.all(surface >= 0.0)


def test_fno_forward_shape() -> None:
    strikes = chebyshev_strike_grid(24)
    maturities = np.linspace(0.1, 1.0, 8)
    model = PricingFNO(width=8, depth=2)
    output = model(build_parameter_field(strikes, maturities))
    assert output.shape == (1, 8, 24)


def test_loss_is_finite_for_decreasing_curve() -> None:
    curve = torch.linspace(10.0, 0.0, 20)[None, :]
    loss = ArbitrageFreeLoss()
    assert torch.isfinite(loss(curve, curve))
