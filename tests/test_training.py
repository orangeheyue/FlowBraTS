import torch

from scripts.train import compute_loss, set_stage
from src.models.network import FlowBraTS
from tests.helpers import TinyM2FAdapter


def make_batch():
    image = torch.randn(1, 4, 8, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.bool)
    return image, target, mask


def test_all_training_stages_have_finite_gradients():
    for stage in ("flow", "utility", "joint", "end_to_end"):
        torch.manual_seed(4)
        model = FlowBraTS(
            TinyM2FAdapter(), channels=8, flow_channels=4, hidden=8, nfe=1
        ).train()
        set_stage(model, stage)
        loss, parts = compute_loss(model, *make_batch(), stage)
        assert torch.isfinite(loss)
        assert parts
        loss.backward()
        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )


def test_end_to_end_unfreezes_backbone_and_innovations():
    model = FlowBraTS(
        TinyM2FAdapter(), channels=8, flow_channels=4, hidden=8, nfe=1
    )
    set_stage(model, "end_to_end")
    assert all(parameter.requires_grad for parameter in model.parameters())

    loss, _ = compute_loss(model, *make_batch(), "end_to_end")
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.flow.parameters())


def test_utility_reference_decoder_does_not_retain_gradients():
    model = FlowBraTS(TinyM2FAdapter(),channels=8,flow_channels=4,hidden=8,nfe=1).train()
    image,target,mask=make_batch()
    teacher=model.extract_teacher(image)
    output=model(image,mask,mode='joint_train',teacher_latent=teacher)
    assert not output['ungated_logits'].requires_grad
    assert not output['baseline_logits'].requires_grad
    assert output['logits'].requires_grad
    segmentation_loss=__import__('src.losses',fromlist=['segmentation_loss']).segmentation_loss
    segmentation_loss(output['logits'],target).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.flow.parameters())
