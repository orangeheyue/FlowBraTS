import pytest
import torch

from src.checkpoint import full_state, innovation_state, load_checkpoint, load_innovations
from src.models.network import FlowBraTS
from tests.helpers import TinyM2FAdapter


def test_innovation_checkpoint_is_strict(tmp_path):
    model = FlowBraTS(TinyM2FAdapter(), channels=8, flow_channels=4, hidden=8, nfe=1)
    path = tmp_path / "valid.pt"
    torch.save({"model": innovation_state(model)}, path)
    load_innovations(model, path)

    broken = innovation_state(model)
    broken.pop(next(iter(broken)))
    path = tmp_path / "broken.pt"
    torch.save({"model": broken}, path)
    with pytest.raises(RuntimeError, match="missing"):
        load_innovations(model, path)


def test_full_checkpoint_restores_backbone(tmp_path):
    model = FlowBraTS(TinyM2FAdapter(), channels=8, flow_channels=4, hidden=8, nfe=1)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "full.pt"
    torch.save({"model": full_state(model), "checkpoint_type": "full"}, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    load_checkpoint(model, path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])
