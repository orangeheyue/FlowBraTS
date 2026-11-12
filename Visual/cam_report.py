"""Quantify cached CAMs and render matched model/scenario comparisons."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cam_worker import CLASSES, cache_path
from common import MODEL_LABELS, SCENARIOS, load_case, atomic_json
from render import _font, _centred_text, _fit_panel, display_crop, normalise_mri, orient


def pearson(a, b, brain):
    a, b = a[brain].astype(np.float64), b[brain].astype(np.float64)
    if a.size < 2:
        return None
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.dot(a,b)/denominator,-1,1)) if denominator > 0 else None


def dice(a, b):
    denominator = int(a.sum())+int(b.sum())
    return float(2*np.logical_and(a,b).sum()/denominator) if denominator else None


def measurements(heat, baseline, target, brain, pred, full_pred, tumor=None):
    mass = float(heat[brain].sum(dtype=np.float64))
    roi = (target if tumor is None else tumor) & brain
    inside = float(heat[roi].sum(dtype=np.float64)/mass) if mass > 0 and roi.any() else None
    class_inside = float(heat[target & brain].sum(dtype=np.float64)/mass) if mass > 0 and target.any() else None
    dsc = dice(pred, target)
    full_dsc = dice(full_pred,target)
    return dict(inside_activation_ratio=inside,
                outside_activation_ratio=1-inside if inside is not None else None,
                class_inside_activation_ratio=class_inside,
                stability_pearson=pearson(heat,baseline,brain),
                slice_class_dice=dsc,
                slice_dice_drop=full_dsc-dsc if full_dsc is not None and dsc is not None else None,
                positive_cam_mass=mass, gt_class_pixels=int(target.sum()),
                valid_cam=bool(mass>0 and roi.any()))


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def colourmap(t):
    # Fixed blue -> cyan -> yellow -> red scale; same limits across 20 panels.
    anchors=np.array([[0.05,0.05,0.45],[0,0.65,1],[1,0.95,0],[0.8,0,0]])
    positions=np.linspace(0,1,len(anchors))
    return np.stack([np.interp(t,positions,anchors[:,c]) for c in range(3)],-1)


def render_heatmaps(year,case,class_index,all_cams,brain,volume,seg,out):
    _,class_name=CLASSES[class_index]
    crop=display_crop(volume,case['slice'])
    background=orient(np.asarray(volume[:,:,case['slice'],0])[crop])
    gt=orient(np.asarray(seg[:,:,case['slice']])[crop])
    gray=normalise_mri(background)
    label=CLASSES[class_index][0]
    target=gt==label
    # A one-pixel inner GT boundary.
    interior=target.copy()
    interior[1:,:] &= target[:-1,:];interior[:-1,:] &= target[1:,:]
    interior[:,1:] &= target[:,:-1];interior[:,:-1] &= target[:,1:]
    boundary=target & ~interior
    densities={}
    for key,cam in all_cams.items():
        values=cam[class_index]*brain
        mass=values.sum(dtype=np.float64)
        densities[key]=values/mass if mass>0 else np.zeros_like(values)
    positive=np.concatenate([v[v>0] for v in densities.values()])
    vmax=float(np.percentile(positive,99)) if positive.size else 1.
    tile_w,tile_h,gap=275,265,8
    top,footer=114,110
    width=6*tile_w+5*gap+24
    height=5*tile_h+4*gap+top+footer
    canvas=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(canvas)
    suffix=' | GT class absent' if not target.any() else ''
    _centred_text(draw,(0,0,width,60),f'BraTS{year} | {case["case_id"]} | slice {case["slice"]} | {class_name}{suffix}',_font(23))
    headings=['Available modalities','FLAIR + GT outline',*MODEL_LABELS.values()]
    for col,title in enumerate(headings):
        x=12+col*(tile_w+gap)
        _centred_text(draw,(x,60,x+tile_w,top),title,_font(17,True))
    for row,(scenario,_,mask) in enumerate(SCENARIOS):
        y=top+row*(tile_h+gap)
        available=[name for name,flag in zip(('FLAIR','T1ce','T1','T2'),mask) if flag]
        _centred_text(draw,(12,y,12+tile_w,y+tile_h),' + '.join(available[:2])+ ('\n'+' + '.join(available[2:]) if len(available)>2 else ''),_font(20,True))
        base=np.repeat(gray[...,None],3,-1);base[boundary]=1
        canvas.paste(_fit_panel((base*255).astype('uint8'),(tile_w,tile_h)),(12+tile_w+gap,y))
        for col,model in enumerate(MODEL_LABELS,start=2):
            density=orient(densities[(model,scenario)][crop])
            values=np.clip(density/vmax,0,1)
            rgb=np.repeat(gray[...,None],3,-1)
            alpha=0.7*(values>0)[...,None]
            rgb=rgb*(1-alpha)+colourmap(values)*alpha
            rgb[boundary]=1
            canvas.paste(_fit_panel((rgb*255).astype('uint8'),(tile_w,tile_h)),(12+col*(tile_w+gap),y))
    bar_y=height-footer+18
    bar=(colourmap(np.linspace(0,1,500)[None,:])*255).astype('uint8')
    canvas.paste(Image.fromarray(bar).resize((500,18)),((width-500)//2,bar_y))
    _centred_text(draw,(0,bar_y+22,width,bar_y+49),'Low relative activation                       High relative activation',_font(14))
    _centred_text(draw,(0,height-39,width,height-5),'White: GT class outline. CAMs have unit mass within brain; shared p99 display scale. FLAIR is display only.',_font(14))
    folder=out/f'BraTS{year}';folder.mkdir(parents=True,exist_ok=True)
    stem=folder/f'{case["case_id"]}_slice_{case["slice"]}_{class_name}_CAM'
    canvas.save(stem.with_suffix('.png'),dpi=(240,240))
    canvas.save(stem.with_suffix('.pdf'),resolution=240.)


def report(manifest,years,cache,out):
    manifest=json.loads(Path(manifest).read_text());rows=[]
    for year in years:
        for case in manifest['datasets'][str(year)]['cases']:
            volume,seg=load_case(year,case['case_id'])
            z=case['slice']; gt=np.asarray(seg[:,:,z])
            brain=np.any(np.abs(volume[:,:,z,:])>1e-8,axis=-1) | (gt>0)
            cams={};preds={}
            for model in MODEL_LABELS:
                for scenario,_,_ in SCENARIOS:
                    path=cache_path(cache,year,model,case['case_id'],scenario)
                    with np.load(path) as f:
                        cams[model,scenario]=f['cam'];preds[model,scenario]=f['prediction']
            for model in MODEL_LABELS:
                for scenario,_,_ in SCENARIOS:
                    for idx,(label,name) in enumerate(CLASSES):
                        metrics=measurements(cams[model,scenario][idx],cams[model,'all_present'][idx],
                                             gt==label,brain,preds[model,scenario]==label,
                                             preds[model,'all_present']==label,tumor=gt>0)
                        rows.append(dict(year=year,case_id=case['case_id'],slice=z,model=model,
                                         scenario=scenario,target_class=name,**metrics))
            for idx in range(3):
                render_heatmaps(year,case,idx,cams,brain,volume,seg,out/'figures')
            print(f'Rendered 3 classes: BraTS{year} {case["case_id"]}',flush=True)
    write_csv(out/'metrics_detail.csv',rows)
    summary=[]
    metrics=['inside_activation_ratio','outside_activation_ratio','stability_pearson',
             'class_inside_activation_ratio','slice_class_dice','slice_dice_drop']
    for year in years:
        for model in MODEL_LABELS:
            for scenario,_,_ in SCENARIOS:
                for _,name in CLASSES:
                    group=[r for r in rows if r['year']==year and r['model']==model and r['scenario']==scenario and r['target_class']==name]
                    item=dict(year=year,model=model,scenario=scenario,target_class=name,n_cases=len(group))
                    for metric in metrics:
                        vals=[r[metric] for r in group if r[metric] is not None]
                        item[metric+'_mean']=float(np.mean(vals)) if vals else None
                        item[metric+'_std']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                        item[metric+'_n_valid']=len(vals)
                    summary.append(item)
    write_csv(out/'metrics_summary.csv',summary)
    atomic_json(out/'COMPLETE.json',dict(years=years,predictions=len(rows)//3,
                                      detail_rows=len(rows),summary_rows=len(summary),
                                      figures=len(years)*5*3,
                                      valid_cam_rows=sum(r['valid_cam'] for r in rows),
                                      absent_gt_rows=sum(r['gt_class_pixels']==0 for r in rows),
                                      zero_cam_rows=sum(r['positive_cam_mass']==0 for r in rows),
                                      inside_outside_region='WT GT within brain',
                                      stability='Pearson within brain vs same-case full modalities'))


if __name__=='__main__':
    p=argparse.ArgumentParser();here=Path(__file__).resolve().parent
    p.add_argument('--manifest',type=Path,default=here/'selection.json')
    p.add_argument('--years',type=int,nargs='+',default=[2018,2020])
    p.add_argument('--cache-root',type=Path,default=here/'cam_cache')
    p.add_argument('--output-root',type=Path,default=here/'cam_outputs')
    a=p.parse_args();report(a.manifest,a.years,a.cache_root,a.output_root)
