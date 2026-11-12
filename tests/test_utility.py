import torch

from src.losses import safety_loss, utility_target
from src.models.utility import CounterfactualUtility


def test_utility_target_sign_matches_improvement():
    target = torch.ones(1, 2, 2, 2, dtype=torch.long)
    baseline = torch.zeros(1, 4, 2, 2, 2)
    improved = baseline.clone()
    improved[:, 1] = 10
    utility = utility_target(baseline, improved, target)
    assert utility[0, 0] > 0
    assert utility[0, 1] > 0


def test_gate_shapes_and_bounds():
    module = CounterfactualUtility(8, 8)
    observed = torch.randn(2, 8, 2, 2, 2)
    residual = torch.randn_like(observed)
    adjacency = torch.softmax(torch.randn(2, 4, 3), dim=1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    utility, std, region_gate, channel_gate = module(observed, residual, adjacency, mask)
    assert utility.shape == std.shape == region_gate.shape == (2, 3)
    assert channel_gate.shape == (2, 8, 1, 1, 1)
    assert torch.all((region_gate >= 0) & (region_gate <= 1))
    assert torch.all((channel_gate >= 0) & (channel_gate <= 1))


def test_safety_loss_is_zero_when_generated_is_better():
    target = torch.ones(1, 2, 2, 2, dtype=torch.long)
    baseline = torch.zeros(1, 4, 2, 2, 2)
    final = baseline.clone()
    final[:, 1] = 10
    assert safety_loss(final, baseline, target).item() == 0.0
