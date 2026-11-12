import torch

from src.models.flow import ConditionalResidualFlow
from src.models.graph import ModalityRegionGraph


def test_flow_loss_and_euler_sample_are_differentiable():
    torch.manual_seed(1)
    observed = torch.randn(2, 8, 2, 2, 2)
    teacher = torch.randn_like(observed)
    mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    graph = ModalityRegionGraph(8, 8)
    adjacency, need = graph(observed, mask)
    flow = ConditionalResidualFlow(8, 8, flow_channels=4)
    loss = flow.loss(observed, teacher, adjacency, need, mask)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in flow.parameters())
    sample = flow.sample(observed, adjacency.detach(), need.detach(), mask, nfe=2,
                         deterministic=True)
    assert sample.shape == observed.shape
    assert torch.isfinite(sample).all()
