"""Real GPU regression check; smoke weights never constitute formal results."""
import argparse
import json
from pathlib import Path
import torch
from src.config import build_model,load_config,seed_everything
from src.checkpoint import load_checkpoint
from src.data.m2f_data import build_dataset
from src.data.protocol import MASKS
from scripts.train import compute_loss,set_stage


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/v21_brats2020.yaml')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='outputs/v21_real_check.json')
    args=p.parse_args();config=load_config(args.config);seed_everything(config['seed'])
    torch.set_num_threads(4)
    model=build_model(config).cuda();load_checkpoint(model,args.checkpoint)
    assert getattr(model,'version',None)=='v21'
    dataset=build_dataset(config['m2ftrans']['root'],config['data']['root'],config['data']['split_dir'],'val',config['data'].get('split_index'))
    image,target,_=dataset[0]
    image=image[None,:,:80,:80,:80].cuda();target=target[None,:80,:80,:80].cuda()
    mask=torch.ones(1,4,dtype=torch.bool,device='cuda')
    set_stage(model,'end_to_end');model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss,_=compute_loss(model,image,target,mask,'end_to_end')
    assert torch.isfinite(loss);loss.backward()
    checks={}
    for name,module in [('flow',model.flow),*model.refinement.items()]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        checks[name]=bool(grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0)
        assert checks[name],name
    assert model.flow.to_latent.weight.grad.abs().sum()>0
    for refiner in model.refinement.values():assert refiner.spatial.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True);model.eval()
    del loss
    with torch.no_grad():
        full=model(image,mask)
        base=model(image,mask,mode='baseline')
        delta=float((full['logits']-base['logits']).abs().max())
        assert delta>1e-7 and full['residual'].abs().sum()>0
        shapes={name:list(g.shape) for name,g in full['spatial_gates'].items()}
        for g in full['spatial_gates'].values():assert g.flatten(2).std(2).max()>0
        for bits in MASKS:
            result=model(image,torch.tensor([bits],device='cuda'))
            assert result['logits'].shape==(1,4,80,80,80)
            assert torch.isfinite(result['logits']).all()
        missing=torch.tensor([[1,0,1,0]],dtype=torch.bool,device='cuda')
        changed=image.clone();changed[:,1]=999;changed[:,3]=-999
        torch.testing.assert_close(model(image,missing)['logits'],model(changed,missing)['logits'])
    report=dict(passed=True,formal_evaluation=False,full_modal_gradients=checks,
                full_modal_max_logit_change=delta,spatial_gate_shapes=shapes,
                inference_masks_checked=15,missing_channel_invariance=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
