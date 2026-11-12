"""Original M2FTrans sliding-window/metrics and validation-only model selection."""
import ast
import csv
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from medpy.metric import hd95
from src.data.m2f_data import build_dataset


def reference(config):
    root=Path(config['m2ftrans']['root'])
    names={'compute_BraTS_HD95','cal_hd95','softmax_output_dice_class4',
           'AverageMeter','test_dice_hd95_softmax'}
    tree=ast.parse((root/'predict.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    ns=dict(torch=torch,np=np,hd95=hd95,csv=csv,logging=logging,time=time,patch_size=80)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/'predict.py'),'exec'),ns)
    for node in ast.parse((root/'eval.py').read_text()).body:
        if isinstance(node,ast.Assign):
            for t in node.targets:
                if isinstance(t,ast.Name) and t.id in ('masks_test','mask_name'):
                    ns[t.id]=ast.literal_eval(node.value)
    return ns


def selected_masks(protocol,split,names=None):
    pairs=list(zip(protocol['masks_test'][::-1],protocol['mask_name'][::-1]))
    if names is None:return pairs
    if split=='test':raise ValueError('Test evaluation must include all 15 combinations')
    if not names or len(names)!=len(set(names)):
        raise ValueError('Validation mask names must be nonempty and unique')
    unknown=set(names)-set(protocol['mask_name'])
    if unknown:raise ValueError(f'Unknown validation masks: {sorted(unknown)}')
    return [(mask,name) for mask,name in pairs if name in names]


class Wrapper(nn.Module):
    def __init__(self,model,mode):
        super().__init__();self.model=model;self.mode=mode;self.is_training=False
    def forward(self,x,mask):
        if self.mode=='official_baseline':
            # The original model receives unmodified inputs and its native mask.
            return self.model.backbone.backbone(x,mask).softmax(1)
        return self.model(x,mask,mode=self.mode)['logits'].softmax(1)


def evaluate(model,config,split,folder,mode='infer',mask_names=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    protocol=reference(config)
    dataset=build_dataset(config['m2ftrans']['root'],config['data']['root'],config['data']['split_dir'],split,config['data'].get('split_index'))
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
    wrapped=nn.DataParallel(Wrapper(model,mode)).eval()
    results=[]
    with torch.no_grad():
        for mask,name in selected_masks(protocol,split,mask_names):
            path=folder/f'{name}.csv'
            # Complete combinations can be reused after interruption in the same
            # checkpoint-specific directory; preserve incomplete output for audit.
            rows=list(csv.reader(path.open())) if path.exists() else []
            if len(rows)==len(dataset)+1 and all(len(row)==8 for row in rows):
                d=protocol['AverageMeter']();h=protocol['AverageMeter']()
                for row in np.asarray(rows[1:],dtype=float):
                    if not np.isfinite(row).all():raise ValueError(path)
                    d.update(row[:4].astype(np.float32));h.update(row[4:])
                dice,distances=d.avg,h.avg
            else:
                if path.exists():path.rename(path.with_suffix(f'.partial.{time.time_ns()}'))
                with path.open('w') as f:
                    csv.writer(f).writerow(['WT Dice','TC Dice','ET Dice','ETPro Dice','WT HD95','TC HD95','ET HD95','ETPro HD95'])
                logging.info('Full %s evaluation: %s (%d cases)',split,name,len(dataset))
                dice,distances=protocol['test_dice_hd95_softmax'](
                    loader,wrapped,dataname='/home/sjj/MMMSeg/BraTS/BRATS2020',
                    feature_mask=mask,mask_name=name,csv_name=str(path))
            results.append(dict(mask=name,channels=mask,dice=np.asarray(dice).tolist(),hd95=np.asarray(distances).tolist()))
    result=dict(split=split,mode=mode,cases=dataset.names,combinations=results,
                selection_protocol='all_15' if mask_names is None else 'fixed_representative_masks',
                mean_dice=np.mean([r['dice'] for r in results],axis=0).tolist(),
                mean_hd95=np.mean([r['hd95'] for r in results],axis=0).tolist(),
                reference_sha256=hashlib.sha256((Path(config['m2ftrans']['root'])/'predict.py').read_bytes()).hexdigest())
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def better(candidate,best,dice_delta=1e-4,hd_delta=.01,selection=None):
    if (selection or {}).get('metric') == 'unime_composite':
        score=float(np.mean(candidate['mean_dice']))
        if not np.isfinite(score):raise ValueError('Invalid validation Dice')
        return best is None or score > float(np.mean(best['mean_dice']))
    d=float(np.mean(candidate['mean_dice'][:3]));h=float(np.mean(candidate['mean_hd95'][:3]))
    if not np.isfinite([d,h]).all():raise ValueError('Invalid validation metrics')
    if best is None:return True
    selection=selection or {}
    if selection.get('metric')=='et_guarded':
        minimum=selection.get('minimum_dice',[0.,0.])
        if len(minimum)!=2:raise ValueError('minimum_dice must contain WT and TC thresholds')
        if candidate['mean_dice'][0]<minimum[0] or candidate['mean_dice'][1]<minimum[1]:return False
        et=float(candidate['mean_dice'][2]);bet=float(best['mean_dice'][2])
        ehd=float(candidate['mean_hd95'][2]);behd=float(best['mean_hd95'][2])
        return et>bet+dice_delta or (abs(et-bet)<=dice_delta and ehd<behd-hd_delta)
    bd=float(np.mean(best['mean_dice'][:3]));bh=float(np.mean(best['mean_hd95'][:3]))
    return d>bd+dice_delta or (abs(d-bd)<=dice_delta and h<bh-hd_delta)


def baseline_cache(config,generated):
    """Reuse only a completed official baseline with matching cases/protocol/source."""
    cache=config.get('comparison',{}).get('cache_summary')
    if not cache or not Path(cache).is_file():return None
    path=Path(cache)
    source_config=path.parent.parent/'config.json'
    if not source_config.exists():return None
    original=json.loads(source_config.read_text())
    expected=config.get('comparison',{}).get('checkpoint',config['m2ftrans']['checkpoint'])
    if expected is None or original['m2ftrans'].get('checkpoint') is None:return None
    if Path(original['m2ftrans']['checkpoint']).resolve()!=Path(expected).resolve():return None
    cached=json.loads(path.read_text())
    if (cached.get('split')!='test' or cached.get('mode')!='official_baseline'
        or cached.get('reference_sha256')!=generated.get('reference_sha256')
        or cached.get('cases')!=generated.get('cases')
        or [r['mask'] for r in cached.get('combinations',[])]!=[r['mask'] for r in generated['combinations']]):return None
    if len(cached['combinations'])!=15:return None
    for key in ['dice','hd95']:
        if not np.isfinite([r[key] for r in cached['combinations']]).all():return None
    return dict(cached,reused_from=str(path))
