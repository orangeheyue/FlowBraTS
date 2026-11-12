import numpy as np
import torch

from src.config import load_config
from src.data.et_dataset import ETPatchDataset, crop_start
from src.models.et_head import ETResidualHead
from src.losses import et_tversky_loss


def test_et_head_preserves_wt_tc_and_has_no_missing_modality_leakage():
    torch.manual_seed(31)
    head=ETResidualHead()
    logits=torch.randn(2,4,8,8,8,requires_grad=True)
    image=torch.randn(2,4,8,8,8)
    mask=torch.tensor([[True,False,True,False],[True,True,True,True]])
    out=head(logits,image,mask)
    torch.testing.assert_close(out.softmax(1),logits.softmax(1),rtol=1e-5,atol=1e-6)
    with torch.no_grad():
        head.offset.weight.normal_(std=.1)
    out=head(logits,image,mask)
    before,after=logits.softmax(1),out.softmax(1)
    torch.testing.assert_close(after[:,1]+after[:,3],before[:,1]+before[:,3])
    torch.testing.assert_close(after[:,1:].sum(1),before[:,1:].sum(1))
    modified=image.clone();modified[0,1]=999;modified[0,3]=-999
    torch.testing.assert_close(out,head(logits,modified,mask))
    out[:,3].mean().backward()
    assert head.offset.weight.grad.abs().sum()>0
    assert torch.isfinite(logits.grad).all()


def test_lesion_patches_include_small_et_and_use_only_training_ids(tmp_path):
    for directory in ['vol','seg','splits']:(tmp_path/directory).mkdir()
    (tmp_path/'splits/train.txt').write_text('training_case\n')
    (tmp_path/'splits/val.txt').write_text('validation_must_not_load\n')
    seg=np.zeros((20,20,20),dtype=np.uint8);seg[1,18,3]=3
    np.save(tmp_path/'seg/training_case_seg.npy',seg)
    np.save(tmp_path/'vol/training_case_vol.npy',np.zeros((20,20,20,4),np.float32))
    dataset=ETPatchDataset(tmp_path,tmp_path/'splits',patch_size=8,
                           et_probability=1.,tc_probability=0.)
    for _ in range(10):
        x,y,mask,name=dataset[0]
        assert name=='training_case' and (y==3).any()
        assert x.shape==(4,8,8,8) and mask.any()
    for center in [(0,0,0),(19,19,19),(10,10,10)]:
        start=crop_start((20,20,20),8,center)
        assert all(s<=c<s+8 for s,c in zip(start,center))


def test_three_configs_have_matched_seed_data_budget_and_validation_only_selection():
    configs=[load_config(f'configs/et_sweep_{name}.yaml')
             for name in ['a_sampling','b_recall','c_finehead']]
    for c in configs:
        assert c['device']=='cuda:0'
        assert c['train']['epochs']==8 and c['train']['supervised_only']
        assert c['train']['scope']=='decoder' and not c['validation']['auto_test']
        assert c['validation']['masks']==['flairt1cet1t2']
        assert c['validation']['initial_summary'].endswith('val_epoch_0020/summary.json')
    assert configs[0]['seed']==configs[1]['seed']==configs[2]['seed']
    assert 'loss' not in configs[0] and configs[1]['loss']['et_tversky_beta']==.7
    assert configs[2]['model']['et_head']


def test_recall_tversky_penalizes_false_negatives_more_than_false_positives():
    target=torch.full((1,2,2,2),3,dtype=torch.long)
    logits=torch.zeros((1,4,2,2,2))
    recall=et_tversky_loss(logits,target,alpha=.3,beta=.7)
    precision=et_tversky_loss(logits,target,alpha=.7,beta=.3)
    assert recall>precision
