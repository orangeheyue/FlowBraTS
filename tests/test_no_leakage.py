import torch

from src.models.network import FlowBraTS
from tests.helpers import TinyM2FAdapter


def test_replacing_missing_modalities_cannot_change_prediction():
    torch.manual_seed(2)
    model = FlowBraTS(TinyM2FAdapter(), channels=8, flow_channels=4, hidden=8, nfe=1).eval()
    image = torch.randn(1, 4, 8, 8, 8)
    mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    changed = image.clone()
    changed[:, 1] = 1000 * torch.randn_like(changed[:, 1])
    changed[:, 3] = 1000 * torch.randn_like(changed[:, 3])
    with torch.no_grad():
        first = model(image, mask)["logits"]
        second = model(changed, mask)["logits"]
    torch.testing.assert_close(first, second)


def test_full_modal_input_bypasses_generation():
    model = FlowBraTS(TinyM2FAdapter(), channels=8, flow_channels=4, hidden=8, nfe=1).eval()
    image = torch.randn(2, 4, 8, 8, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        output = model(image, mask)
    assert torch.count_nonzero(output["residual"]) == 0
    torch.testing.assert_close(output["logits"], output["baseline_logits"])
