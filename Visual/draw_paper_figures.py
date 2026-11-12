"""Reproducible, code-grounded FlowBraTS paper diagrams (PNG + SVG + PDF).

Uses Matplotlib drawing primitives, real BraTS slices and existing prediction
cache. No generative imaging, GPU inference, or invented performance numbers.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, PathPatch, Polygon
from matplotlib.path import Path as MPath
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'preproced_dataset/BraTS2020'
CASE = 'HG_BraTS20_Training_097'
SLICE = 67

C = dict(ink='#25384B', muted='#66788A', line='#B6C2CD', panel='#F6F8FA',
         blue='#377DAB', blue_light='#EDF4FA', teal='#1D897E', teal_light='#EBF6F3',
         orange='#CB7D2F', orange_light='#FEF4E7', purple='#7B69A6',
         purple_light='#F3EFF9', red='#CB6472', red_light='#FCEFF1',
         ed='#39AE86', et='#EFCB59', ncr='#D97079')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 17,
                     'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def text(ax, x, y, value, size=17, color=None, weight='normal', ha='center', va='center', max_width=None, **kw):
    artist=ax.text(x, y, value, fontsize=size, color=color or C['ink'], fontweight=weight,
                   ha=ha, va=va, linespacing=1.35, zorder=10, **kw)
    # Fit using actual font metrics, rather than character counts. The smallest
    # containing diagram box determines the available width unless overridden.
    if max_width is None:
        candidates=[b for b in getattr(ax,'_diagram_boxes',[]) if b[0]<=x<=b[0]+b[2] and b[1]<=y<=b[1]+b[3]]
        if candidates:
            left,_,width,_=min(candidates,key=lambda b:b[2]*b[3])
            max_width=(2*min(x-left,left+width-x) if ha=='center' else left+width-x if ha=='left' else x-left)-18
    if max_width and max_width>0:
        available=abs(ax.transData.transform((max_width,0))[0]-ax.transData.transform((0,0))[0])
        measured=artist.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
        if measured>available:
            artist.set_fontsize(size*available/measured*.97)
    return artist


def box(ax, x, y, w, h, fill='white', edge=None, radius=10, lw=1.1, dashed=False):
    patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                         facecolor=fill,edgecolor=edge or C['line'],linewidth=lw,
                         linestyle=(0,(5,4)) if dashed else '-',zorder=1)
    ax.add_patch(patch)
    if not hasattr(ax,'_diagram_boxes'):ax._diagram_boxes=[]
    ax._diagram_boxes.append((x,y,w,h))
    return patch


def arrow(ax, points, color=None, lw=1.4, dashed=False, head=12, zorder=4):
    color=color or C['ink']
    for start,end in zip(points[:-2],points[1:-1]):
        ax.plot([start[0],end[0]],[start[1],end[1]],color=color,lw=lw,
                ls=(0,(4,3)) if dashed else '-',solid_capstyle='round',zorder=zorder)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                                mutation_scale=head,lw=lw,color=color,
                                linestyle=(0,(4,3)) if dashed else '-',zorder=zorder))


def dot(ax,x,y,color,r=3):
    ax.add_patch(Circle((x,y),r,facecolor=color,edgecolor='none',zorder=6))


def normalise(a):
    valid=a[np.abs(a)>1e-8]
    if not valid.size:return np.zeros_like(a)
    lo,hi=np.percentile(valid,(1,99))
    return np.clip((a-lo)/max(hi-lo,1e-6),0,1)


def anatomy():
    volume=np.load(DATA/'vol'/f'{CASE}_vol.npy',mmap_mode='r')
    seg=np.load(DATA/'seg'/f'{CASE}_seg.npy',mmap_mode='r')
    brain=np.any(np.abs(volume[:,:,SLICE,:])>1e-8,axis=-1)
    coords=np.argwhere(brain)
    low=np.maximum(coords.min(0)-3,0);high=np.minimum(coords.max(0)+4,brain.shape)
    crop=(slice(low[0],high[0]),slice(low[1],high[1]))
    images=[np.rot90(normalise(np.asarray(volume[:,:,SLICE,i])[crop])) for i in range(4)]
    labels=np.rot90(np.asarray(seg[:,:,SLICE])[crop])
    return images,labels,crop


def image_tile(ax,image,x,y,w,h,label=None,missing=False,label_size=14):
    if missing:
        box(ax,x,y,w,h,C['red_light'],C['red'],radius=4,dashed=True)
        ax.plot([x+w*.32,x+w*.68],[y+h*.3,y+h*.7],color=C['red'],lw=1.7,zorder=5)
        ax.plot([x+w*.32,x+w*.68],[y+h*.7,y+h*.3],color=C['red'],lw=1.7,zorder=5)
    else:
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#0B1015',edgecolor='none',zorder=2))
        ax.imshow(image,cmap='gray',vmin=0,vmax=1,extent=(x,x+w,y,y+h),
                  aspect='auto',interpolation='bilinear',zorder=3)
    if label:text(ax,x+w/2,y-16,label,size=label_size,color=C['red'] if missing else C['ink'])


def feature_stack(ax,x,y,w=65,h=75,color=None,layers=3):
    color=color or C['blue']
    for i in reversed(range(layers)):
        ax.add_patch(Rectangle((x+i*7,y+i*5),w,h,facecolor='white',edgecolor=color,lw=1.2,zorder=3))
        ax.add_patch(Rectangle((x+i*7,y+i*5),w,h,facecolor=color,alpha=.1+i*.025,edgecolor='none',zorder=3))
    for fx,fy in ((.35,.62),(.65,.36),(.58,.71)):
        ax.add_patch(Circle((x+w*fx,y+h*fy),w*.1,facecolor=color,alpha=.36,edgecolor='none',zorder=4))


def panel_label(ax,x,y,letter,title,color=None,size=20):
    text(ax,x,y,f'({letter})',size=size,weight='bold',ha='left',color=color)
    text(ax,x+48,y,title,size=size,weight='bold',ha='left',color=color)


def save(fig,stem):
    for suffix in ('png','svg','pdf'):
        fig.savefig(HERE/f'{stem}.{suffix}',dpi=400,facecolor='white',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


def motivation():
    images,labels,_=anatomy()
    fig,ax=plt.subplots(figsize=(16,7.2));ax.set(xlim=(0,1600),ylim=(0,720));ax.axis('off')
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
    text(ax,800,686,'From incomplete observations to task-aware residual correction',size=25,weight='bold')
    panels=((20,490,'a','Region-dependent evidence',C['teal_light']),
            (535,490,'b','Task-aware residual Flow',C['orange_light']),
            (1050,530,'c','Utility-aware refinement',C['purple_light']))
    for x,w,letter,title,fill in panels:
        box(ax,x,99,w,532,fill,edge='#D9E1E7',radius=14)
        panel_label(ax,x+18,602,letter,title,size=18)

    # (a) A missing T1ce example; all graph connections exist, priors are emphasis.
    text(ax,265,558,'Which regional evidence is missing?',size=17,weight='bold',color=C['teal'])
    node_x=[83,203,323,443]
    for index,(name,x) in enumerate(zip(('FLAIR','T1ce','T1','T2'),node_x)):
        image_tile(ax,images[index],x-45,414,90,114,name,missing=index==1,label_size=15)
    text(ax,265,368,'Missing-aware modality–region dependencies',size=15)
    top_y,bottom_y=327,224
    target_x=[108,264,420]
    prior=np.array([[0,0,2],[2,3,0],[2,0,0],[0,0,2]])
    for j,x in enumerate(node_x):
        for k,tx in enumerate(target_x):
            strong=prior[j,k]>0;missing=j==1
            colour=C['red'] if missing and strong else C['teal'] if strong else '#C6D6D4'
            ax.plot([x,tx],[top_y-8,bottom_y+12],color=colour,lw=2 if strong else .65,
                    alpha=1 if strong else .65,ls=(0,(4,3)) if missing else '-',zorder=2)
    for j,x in enumerate(node_x):
        ax.add_patch(Circle((x,top_y),10,facecolor='white' if j==1 else C['teal'],
                            edgecolor=C['red'] if j==1 else C['teal'],lw=1.4,zorder=4))
    for x,name,colour in zip(target_x,('NCR/NET','ET','ED'),(C['ncr'],C['et'],C['ed'])):
        ax.add_patch(Circle((x,bottom_y),14,facecolor=colour,edgecolor='white',lw=1.3,zorder=4))
        text(ax,x,192,name,size=15,weight='bold')
    text(ax,265,145,'The evidence gap depends on the missing modality.',size=14)
    text(ax,265,119,'Prior-guided dependencies shown schematically',size=12,color=C['muted'])

    # (b) One conditional residual trajectory, drawn schematically in feature space.
    text(ax,780,558,'What correction should be generated?',size=17,weight='bold',color=C['orange'])
    box(ax,575,482,410,43,'white',C['orange'],radius=7)
    text(ax,780,504,r'Condition: $z$, graph $A$, need $n$, mask $m$',size=14)
    feature_stack(ax,580,343,68,82,C['blue'])
    feature_stack(ax,891,343,68,82,C['orange'])
    for offset in (-24,0,24):
        path=MPath([(675,384+offset),(738,464+offset*.3),(796,314+offset*.3),(875,384+offset)],
                   [MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
        ax.add_patch(PathPatch(path,facecolor='none',edgecolor=C['orange'],lw=1.5,alpha=.85,zorder=4))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=11,lw=1.1,color=C['orange']))
    text(ax,626,306,r'Conditional prior $r_0$',size=15)
    text(ax,930,306,r'Task residual $r_1$',size=15)
    text(ax,780,262,r'$\dot r_t=v_\theta(r_t,t\mid z,A,n,m)$',size=21,color=C['orange'])
    text(ax,780,206,'Full-modal feature gap',size=16)
    text(ax,780,177,'+ segmentation-error target',size=16)
    text(ax,780,144,'Training supervision',size=12,color=C['muted'])
    text(ax,780,119,'Correction is transported in feature space.',size=14)

    # (c) Global conservative utility, then spatial refinement at three scales.
    text(ax,1315,558,'Where should correction be trusted?',size=17,weight='bold',color=C['purple'])
    box(ax,1090,456,450,70,'white',C['purple'],radius=8)
    text(ax,1315,502,'Predicted utility   −   uncertainty penalty',size=16)
    text(ax,1315,472,r'$g=\mathrm{sigmoid}((\hat u-\kappa\sigma-\tau)/T)$',size=18,color=C['purple'])
    for i,name in enumerate(('WT','TC','ET')):
        x=1111+i*146
        box(ax,x,380,124,43,'white','#CBBFDD',radius=5)
        text(ax,x+62,402,name,size=16,weight='bold',color=C['purple'])
    arrow(ax,[(1315,453),(1315,431)],C['purple'])
    text(ax,1315,348,'Global confidence  +  local spatial gates',size=15)
    for i,(res,channels) in enumerate(((5,256),(10,128),(20,64))):
        x=1110+i*148
        feature_stack(ax,x,214,75,82,C['purple'])
        text(ax,x+46,188,rf'${channels}\times {res}^3$',size=16)
    arrow(ax,[(1315,335),(1315,315)],C['purple'])
    text(ax,1315,148,'Selective residual updates at three feature scales',size=14)
    text(ax,1315,120,'One shared Flow conditions all refinement heads.',size=13,color=C['muted'])

    arrow(ax,[(510,360),(529,360)],C['ink'],head=14)
    arrow(ax,[(1025,360),(1044,360)],C['ink'],head=14)
    text(ax,800,59,'FlowBraTS: infer the evidence gap, transport a task residual, and modulate its local effect.',size=18,weight='bold')
    text(ax,800,25,'Conceptual mechanism illustration; trajectories and gates do not represent measured performance.',size=12,color=C['muted'])
    save(fig,'FlowBraTS_motivation')


def architecture():
    images,labels,crop=anatomy()
    fig,ax=plt.subplots(figsize=(18,12.6));ax.set(xlim=(0,1800),ylim=(0,1260));ax.axis('off')
    fig.subplots_adjust(left=.01,right=.99,bottom=.015,top=.99)
    text(ax,900,1228,'FlowBraTS: missing-aware, utility-gated task residual refinement',size=27,weight='bold')

    # Overall inference panel.
    box(ax,20,635,1760,540,C['panel'],'#DAE2E9',radius=12)
    panel_label(ax,40,1146,'a','Inference with available modalities',size=21)
    text(ax,1740,1146,'Residual VelocityNet  ·  Euler-S2',size=15,color=C['muted'],ha='right')

    # A compact true multimodal input example, with T1 excluded.
    for i,(name,image) in enumerate(zip(('FLAIR','T1ce','T1','T2'),images)):
        y=898-i*58
        image_tile(ax,image,48,y,48,51,missing=i==2)
        text(ax,110,y+26,name,size=13,ha='left',color=C['red'] if i==2 else C['ink'])
    text(ax,107,687,r'$X_{\rm obs},\;m$',size=19)
    text(ax,107,659,r'$4\times80^3$',size=14,color=C['muted'])
    arrow(ax,[(170,824),(205,824)])

    box(ax,210,772,230,180,C['blue_light'],C['blue'],radius=10)
    text(ax,325,928,r'M$^2$FTrans backbone',size=19,weight='bold',color=C['blue'])
    for i in range(4):
        box(ax,229+i*31,829,23,63,'white',C['blue'],radius=3,lw=.9)
    arrow(ax,[(353,859),(378,859)],C['blue'],head=9)
    box(ax,382,827,40,65,'white',C['blue'],radius=4)
    text(ax,325,800,'Encoders + fusion',size=14)
    arrow(ax,[(440,851),(478,851)])
    feature_stack(ax,487,807,58,75,C['blue'])
    text(ax,526,783,r'$z:\;256\times5^3$',size=17)
    text(ax,526,758,'Observed fusion latent',size=13,color=C['muted'])

    # Conditioning chain, above the decoder stream.
    box(ax,470,990,260,123,C['teal_light'],C['teal'],radius=9)
    text(ax,600,1084,'DMG',size=22,weight='bold',color=C['teal'])
    text(ax,600,1054,'Modality–region graph',size=16)
    text(ax,600,1023,r'$A\in\mathbb{R}^{4\times3},\;n\in\mathbb{R}^{3}$',size=17)
    arrow(ax,[(523,897),(523,949),(552,949),(552,990)],C['teal'])
    text(ax,495,956,r'$z,m$',size=14,color=C['teal'],ha='right')

    box(ax,785,990,365,123,C['orange_light'],C['orange'],radius=9)
    text(ax,967,1084,'RFM: residual transport',size=18,weight='bold',color=C['orange'])
    text(ax,967,1052,r'$r_0\;\longrightarrow\;r_1\;\longrightarrow\;\Delta z=Q(r_1)$',size=21)
    text(ax,967,1020,'One ODE in 64 × 5³ residual space',size=15)
    arrow(ax,[(730,1054),(785,1054)],C['teal'])
    text(ax,758,1078,r'$A,n,m$',size=13,color=C['teal'])
    arrow(ax,[(559,856),(603,856),(603,965),(850,965),(850,990)],C['blue'])
    text(ax,700,975,r'$P(z)$',size=14,color=C['blue'])

    box(ax,1245,990,360,123,C['purple_light'],C['purple'],radius=9)
    text(ax,1425,1084,'CFG: counterfactual utility',size=18,weight='bold',color=C['purple'])
    text(ax,1425,1052,r'$(\hat u,\sigma)\;\longrightarrow\;g\in(0,1)^3$',size=20)
    text(ax,1425,1020,'Utility − uncertainty; WT / TC / ET',size=14)
    arrow(ax,[(1150,1054),(1245,1054)],C['orange'])
    text(ax,1198,1078,r'$\Delta z$',size=15,color=C['orange'])
    # Graph skip to utility is explicitly visible.
    arrow(ax,[(660,1113),(660,1126),(1375,1126),(1375,1113)],C['teal'],lw=1.1)
    text(ax,1025,1138,r'$A$; utility also conditions on $z$ and $m$',size=12,color=C['muted'])

    # Main decoder stream: only three sites receive refinement.
    sites=[(680,256,5),(900,128,10),(1120,64,20)]
    arrow(ax,[(559,844),(680,844)],C['blue'])
    for i,(x,channels,size) in enumerate(sites):
        box(ax,x,796,170,95,'white',C['blue'],radius=8)
        box(ax,x+8,829,154,54,C['purple_light'],C['purple'],radius=5)
        text(ax,x+85,857,'Spatial\nrefiner',size=14,weight='bold',color=C['purple'])
        text(ax,x+85,813,rf'${channels}\times{size}^3$',size=16)
        if i<2:
            arrow(ax,[(x+170,844),(x+220,844)],C['blue'])
            text(ax,x+195,872,'↑2',size=13,color=C['blue'])
    arrow(ax,[(1290,844),(1342,844)],C['blue'])
    box(ax,1342,796,180,95,C['blue_light'],C['blue'],radius=8)
    text(ax,1432,857,'Decoder + head',size=17,weight='bold',color=C['blue'])
    text(ax,1432,820,r'$40^3\;\rightarrow\;80^3$',size=18)
    arrow(ax,[(1522,844),(1561,844)],C['blue'])

    # Shared generated residual and shared global gate buses.
    arrow(ax,[(965,990),(965,951),(1205,951),(1205,891)],C['orange'],lw=1.6)
    ax.plot([745,1205],[951,951],color=C['orange'],lw=1.6,zorder=3)
    for x in (745,965):
        arrow(ax,[(x,951),(x,891)],C['orange'],lw=1.5);dot(ax,x,951,C['orange'])
    text(ax,1060,966,r'Shared $\Delta z$',size=14,color=C['orange'])
    arrow(ax,[(1490,990),(1490,922),(1225,922),(1225,891)],C['purple'],lw=1.5)
    ax.plot([765,1225],[922,922],color=C['purple'],lw=1.5,zorder=3)
    for x in (765,985):
        arrow(ax,[(x,922),(x,891)],C['purple'],lw=1.4);dot(ax,x,922,C['purple'])
    text(ax,1394,937,r'Shared global gate $g$',size=14,color=C['purple'])

    # Context/skips, visually separate from correction buses.
    arrow(ax,[(326,772),(326,719),(1432,719),(1432,796)],C['blue'],lw=1.05)
    for x in (985,1205):
        arrow(ax,[(x,719),(x,796)],C['blue'],lw=1.05)
    text(ax,970,693,r'M$^2$FTrans skip / context features',size=14,color=C['blue'])

    pred_path=HERE/'cache/BraTS2020/flowbrats'/CASE/'t1_missing.npy'
    if not pred_path.exists():
        raise FileNotFoundError(f'The architecture output requires a real prediction: {pred_path}')
    pred=np.rot90(np.load(pred_path)[crop])
    rgb=np.repeat(images[0][...,None],3,-1)
    for label,color in ((1,C['ncr']),(2,C['ed']),(3,C['et'])):
        import matplotlib.colors as mcolors
        region=pred==label;rgb[region]=.25*rgb[region]+.75*np.array(mcolors.to_rgb(color))
    ax.imshow(rgb,extent=(1568,1739,768,933),aspect='auto',zorder=4)
    text(ax,1653,743,r'Segmentation $\hat y$',size=17,weight='bold')
    text(ax,1653,715,'BG / NCR-NET / ED / ET',size=12,color=C['muted'])
    text(ax,1653,667,'Solid arrows: inference',size=12,color=C['muted'])

    # Flow detail.
    box(ax,20,248,847,344,C['orange_light'],'#E9D7C1',radius=12)
    panel_label(ax,42,559,'b','Conditional residual transport',C['orange'],size=21)
    text(ax,443,520,r'$o=P(z),\qquad c=[A,n,m,\mathrm{embed}(t)]$',size=20)
    box(ax,52,355,180,104,'white',C['orange'],radius=8)
    text(ax,142,426,'Cond. prior',size=17,weight='bold')
    text(ax,142,390,r'$r_0=\mu(o,A,n,m)$',size=17)
    arrow(ax,[(232,405),(278,405)],C['orange'])
    box(ax,278,355,334,104,'white',C['orange'],radius=8)
    text(ax,445,430,'Residual VelocityNet',size=18,weight='bold',color=C['orange'])
    text(ax,445,396,'1³ Conv → ResBlock ×2 → 1³ Conv',size=13)
    text(ax,445,370,'FiLM: time + graph + need + mask',size=12,color=C['muted'])
    arrow(ax,[(612,405),(659,405)],C['orange'])
    box(ax,659,355,177,104,'white',C['orange'],radius=8)
    text(ax,747,427,'Euler, 2 steps',size=17,weight='bold')
    text(ax,747,389,r'$\Delta z=Q(r_1)$',size=21)
    arrow(ax,[(747,459),(747,474),(445,474),(445,459)],C['orange'],lw=1.05,dashed=True,head=10)
    text(ax,645,486,r'Update $r_t,t$; evaluate $v_\theta$ twice',size=12,color=C['orange'],max_width=365)
    text(ax,444,311,r'$\mathrm{d}r_t/\mathrm{d}t=v_\theta(r_t,t\mid o,A,n,m),\quad t\in[0,1]$',size=20,color=C['orange'])
    text(ax,444,270,'64-channel flow space → 256-channel conditioning residual; zero source noise at inference',size=12,color=C['muted'])

    # Spatial refiner detail: local residual is not the shared Flow output.
    box(ax,892,248,888,344,C['purple_light'],'#DDD5E9',radius=12)
    panel_label(ax,915,559,'c','Utility-conditioned spatial refinement',C['purple'],size=21)
    text(ax,1350,514,r'$g=\mathrm{sigmoid}((\hat u-\kappa\sigma-\tau)/T)$',size=20,color=C['purple'])
    text(ax,944,432,r'$F_s$',size=23)
    text(ax,944,339,r'$\Delta z$',size=23,color=C['orange'])
    box(ax,1021,323,105,41,'white',C['orange'],radius=5)
    text(ax,1073,344,'Proj.\n+ resize',size=12,color=C['orange'])
    arrow(ax,[(974,339),(1021,339)],C['orange'])
    box(ax,1149,380,173,93,'white',C['purple'],radius=7)
    text(ax,1236,441,'Local 3D head',size=17,weight='bold')
    text(ax,1236,410,r'$R_s,\;h_s$',size=21,color=C['purple'])
    arrow(ax,[(974,432),(1149,432)],C['ink'])
    arrow(ax,[(1126,343),(1157,343),(1157,380)],C['orange'])
    box(ax,1367,350,163,64,'white',C['purple'],radius=6)
    text(ax,1449,391,'Spatial gate',size=15,weight='bold',color=C['purple'])
    text(ax,1449,365,r'$\sigma(h_s+\mathrm{logit}(g))$',size=14)
    arrow(ax,[(1322,401),(1367,401)],C['purple'])
    text(ax,1348,421,r'$h_s$',size=13,color=C['purple'])
    arrow(ax,[(1449,321),(1449,350)],C['purple'])
    text(ax,1449,307,'Global gate prior',size=12,color=C['purple'])
    # Product and identity shortcut.
    ax.add_patch(Circle((1590,439),17,facecolor='white',edgecolor=C['purple'],lw=1.3,zorder=6))
    text(ax,1590,439,'×',size=23,color=C['purple'])
    arrow(ax,[(1322,446),(1573,446)],C['purple'])
    text(ax,1454,463,r'Local residual $R_s$',size=13,color=C['purple'])
    arrow(ax,[(1530,382),(1590,382),(1590,422)],C['purple'])
    text(ax,1628,330,r'Channel mixing $\rightarrow G_s$',size=12,color=C['purple'],max_width=260)
    ax.add_patch(Circle((1693,439),17,facecolor='white',edgecolor=C['ink'],lw=1.3,zorder=6))
    text(ax,1693,439,'+',size=23)
    arrow(ax,[(1607,439),(1676,439)],C['purple'])
    arrow(ax,[(1000,432),(1000,487),(1693,487),(1693,456)],C['ink'],lw=1.1)
    arrow(ax,[(1710,439),(1746,439)],C['ink'])
    text(ax,1736,472,r"$F'_s$",size=21)
    text(ax,1328,275,r"$F'_s=F_s+G_s\odot R_s,\qquad s\in\{5^3,10^3,20^3\}$",size=20)

    # Strictly training-only supervision, boxed and dashed.
    box(ax,20,24,1760,183,'#FBFCFD','#9EADB9',radius=10,dashed=True)
    text(ax,44,179,'TRAINING ONLY',size=15,weight='bold',ha='left',color=C['muted'])
    text(ax,1755,179,'Full modalities + GT; shared backbone, detached full-modal features',size=14,ha='right',color=C['muted'])
    ax.plot([590,590],[48,154],color='#D5DDE5',lw=1)
    ax.plot([1165,1165],[48,154],color='#D5DDE5',lw=1)
    text(ax,306,137,'Task-aware Flow target',size=17,weight='bold',color=C['orange'])
    text(ax,306,105,r'$r_1^*=\mathrm{sg}(P(z_{\rm full})-P(z))+e_y$',size=18)
    text(ax,306,69,r'$e_y=0.25\,B\,\mathrm{Pool}(\mathrm{onehot}(y)-\mathrm{sg}(p_{\rm base}))$',size=14)
    text(ax,306,43,'B: fixed class-code projection; sg: stop-gradient',size=12,color=C['muted'])
    text(ax,875,137,'Paired utility supervision',size=17,weight='bold',color=C['purple'])
    text(ax,875,105,r'$u^*=\ell_{\rm reg}(p_{\rm base},y)-\ell_{\rm reg}(p_{\rm ungated},y)$',size=17)
    text(ax,875,69,'Regional utility and heteroscedastic uncertainty',size=14,color=C['muted'])
    text(ax,1468,137,'Joint training objective',size=17,weight='bold')
    text(ax,1468,105,r'$\mathcal{L}_{\rm seg}+\mathcal{L}_{\rm FM}+0.2\mathcal{L}_{\rm utility}$',size=18)
    text(ax,1468,70,r'$+\;0.2\mathcal{L}_{\rm safety}+0.01\mathcal{L}_{\rm graph}$',size=18)
    save(fig,'FlowBraTS_architecture')


if __name__=='__main__':
    motivation();architecture()
    from PIL import Image
    manifest=dict(model_name='FlowBraTS',code_version='v21',
                  checkpoint='outputs/finetune_brats2020/finetune/best.pt',
                  checkpoint_epoch=20,velocity='Residual VelocityNet',solver='Euler',steps=2,
                  source_case=CASE,axial_array_slice=SLICE,backend='Matplotlib vector drawing',
                  sources=['src/models/network_v21.py','src/models/graph.py',
                           'src/models/task_flow.py','src/models/flow.py','src/models/velocity.py',
                           'src/models/utility.py','src/models/refinement.py',
                           'src/models/m2f_adapter_v21.py','scripts/train.py'],files={})
    for stem in ('FlowBraTS_motivation','FlowBraTS_architecture'):
        with Image.open(HERE/f'{stem}.png') as im:
            manifest['files'][stem]=dict(png=f'{stem}.png',pixels=list(im.size),
                                        dpi=list(im.info.get('dpi',(400,400))),
                                        svg=f'{stem}.svg',pdf=f'{stem}.pdf')
    (HERE/'paper_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved FlowBraTS_motivation and FlowBraTS_architecture as PNG / SVG / PDF in',HERE)
