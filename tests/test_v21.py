import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.network import FlowBraTS
from src.models.network_v21 import FlowBraTSV21
from src.checkpoint import full_state,initialize_from_v1,load_checkpoint
from src.data.protocol import apply_mask
from scripts.train import compute_loss,set_stage


class TinyMultiScaleAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Conv3d(4,8,1)
        self.middle=nn.Conv3d(8,4,1)
        self.fine=nn.Conv3d(4,2,1)
        self.output=nn.Conv3d(2,4,1)
    def encode(self,x,mask):
        latent=F.adaptive_avg_pool3d(self.encoder(apply_mask(x,mask)),2)
        return dict(latent=latent,context=None,logits=self.decode(latent,None,mask))
    def decode(self,z,context,mask):
        x=self.middle(F.interpolate(z,scale_factor=2,mode='trilinear',align_corners=False))
        x=self.fine(F.interpolate(x,scale_factor=2,mode='trilinear',align_corners=False))
        return self.output(x)
    def decode_refined(self,z,context,mask,residual,gate,refinement,gated=True):
        records={}
        z,records['bottleneck']=refinement['bottleneck'](z,residual,gate,gated)
        x=self.middle(F.interpolate(z,scale_factor=2,mode='trilinear',align_corners=False))
        x,records['decoder10']=refinement['decoder10'](x,residual,gate,gated)
        x=self.fine(F.interpolate(x,scale_factor=2,mode='trilinear',align_corners=False))
        x,records['decoder20']=refinement['decoder20'](x,residual,gate,gated)
        return self.output(x),records


def model():
    torch.manual_seed(21)
    return FlowBraTSV21(TinyMultiScaleAdapter(),channels=8,flow_channels=4,hidden=8,
                       nfe=1,scale_channels=(8,4,2),refinement_hidden=8)


def batch():
    return torch.randn(1,4,8,8,8),torch.randint(0,4,(1,8,8,8)),torch.ones(1,4,dtype=torch.bool)


def test_full_modal_correction_is_active_at_all_scales():
    m=model().eval();x,y,mask=batch()
    with torch.no_grad():
        out=m(x,mask);baseline=m(x,mask,mode='baseline')
    assert out['residual'].abs().sum()>0
    assert (out['logits']-baseline['logits']).abs().max()>1e-7
    for scale,size in zip(['bottleneck','decoder10','decoder20'],[2,4,8]):
        assert out['spatial_gates'][scale].shape==(1,3,size,size,size)
        assert out['scale_residuals'][scale].abs().sum()>0
        gate=out['spatial_gates'][scale]
        assert (gate>0).all() and (gate<1).all()
        assert gate.flatten(2).std(2).max()>0


def test_full_modal_segmentation_gradient_reaches_flow_and_every_spatial_scale():
    m=model().train();x,y,mask=batch();set_stage(m,'end_to_end')
    loss,_=compute_loss(m,x,y,mask,'end_to_end')
    assert torch.isfinite(loss);loss.backward()
    for module in [m.backbone,m.flow,m.graph,m.utility,*m.refinement.values()]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    for refiner in m.refinement.values():
        assert refiner.spatial.weight.grad.abs().sum()>0
    assert m.flow.to_latent.weight.grad.abs().sum()>0


def test_full_modal_fm_target_is_not_teacher_minus_self_zero():
    m=model();x,y,mask=batch()
    task=m.flow.task_residual_target(torch.zeros(1,4,8,8,8),torch.ones_like(y),(2,2,2))
    assert task.shape==(1,4,2,2,2) and task.abs().sum()>0
    assert not task.requires_grad


def test_supervised_et_training_reaches_flow_without_teacher_or_fm_passes():
    m=model();x,y,mask=batch()
    from src.losses import segmentation_loss
    output=m(x,mask,mode='seg_train')
    assert 'ungated_logits' not in output
    assert float(output['loss_fm'].detach())==0.0
    segmentation_loss(output['logits'],y).backward()
    assert m.flow.to_latent.weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_no_missing_channel_or_label_leakage_at_inference():
    m=model().eval();x,y,mask=batch();mask[:,1]=False;mask[:,3]=False
    changed=x.clone();changed[:,1]=torch.randn_like(changed[:,1])*1000;changed[:,3]=-999
    with torch.no_grad():torch.testing.assert_close(m(x,mask)['logits'],m(changed,mask)['logits'])
    with pytest.raises(ValueError,match='targets are forbidden'):m(x,mask,task_target=y)
    with pytest.raises(ValueError,match='Teacher features are forbidden'):m(x,mask,teacher_latent=torch.zeros(1,8,2,2,2))


def test_v1_transfer_is_explicit_and_v21_checkpoints_are_strict(tmp_path):
    m=model();old=FlowBraTS(TinyMultiScaleAdapter(),channels=8,flow_channels=4,hidden=8,nfe=1)
    path=tmp_path/'v1.pt';torch.save(dict(model=full_state(old),checkpoint_type='full'),path)
    with pytest.raises(RuntimeError):load_checkpoint(m,path)
    report=initialize_from_v1(m,path)
    assert 'flow.task_basis' in report['new_keys']
    assert any(key.startswith('refinement.') for key in report['new_keys'])
    for key,value in old.state_dict().items():torch.testing.assert_close(m.state_dict()[key],value)
    v21=tmp_path/'v21.pt';torch.save(dict(model=full_state(m),checkpoint_type='full'),v21)
    restored=model();load_checkpoint(restored,v21)
    for key,value in m.state_dict().items():torch.testing.assert_close(restored.state_dict()[key],value)


def test_warmup_updates_additions_without_changing_pretrained_backbone():
    m=model();x,y,mask=batch();set_stage(m,'joint')
    before={k:v.detach().clone() for k,v in m.backbone.state_dict().items()}
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.refinement.parameters())
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    loss,_=compute_loss(m,x,y,mask,'joint');loss.backward();optimizer.step()
    assert all(p.grad is None for p in m.backbone.parameters())
    for key,value in before.items():torch.testing.assert_close(m.backbone.state_dict()[key],value,rtol=0,atol=0)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.refinement.parameters())
    set_stage(m,'end_to_end')
    assert all(p.requires_grad for p in m.backbone.parameters())
