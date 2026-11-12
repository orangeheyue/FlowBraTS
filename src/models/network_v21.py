"""FlowBraTSv21: full-modal task flow, multiscale refinement, spatial gates."""
import torch
from torch import nn
from src.data.protocol import validate_mask
from .network import FlowBraTS
from .task_flow import TaskResidualFlow
from .refinement import SpatialResidualRefiner


class FlowBraTSV21(FlowBraTS):
    version='v21'
    MODES = FlowBraTS.MODES + ('seg_train',)
    def __init__(self,backbone,channels=256,flow_channels=64,hidden=64,nfe=2,
                 solver='euler',gate_temperature=.5,utility_threshold=0.,uncertainty_penalty=1.,
                 scale_channels=(256,128,64),refinement_hidden=32,task_scale=.25,
                 task_class_weights=(1.,1.,1.,1.),et_head=False,velocity_type='residual',
                 disable_dmg=False,disable_rfm=False,disable_cfg=False):
        super().__init__(backbone,channels,flow_channels,hidden,nfe,solver,
                         gate_temperature,utility_threshold,uncertainty_penalty)
        self.flow=TaskResidualFlow(channels,hidden,flow_channels,task_scale,
                                   task_class_weights,velocity_type)
        self.disable_dmg=bool(disable_dmg)
        self.disable_rfm=bool(disable_rfm)
        self.disable_cfg=bool(disable_cfg)
        if len(scale_channels)!=3:raise ValueError('Expected bottleneck, decoder10, decoder20 channels')
        self.refinement=nn.ModuleDict({name:SpatialResidualRefiner(c,channels,refinement_hidden)
            for name,c in zip(['bottleneck','decoder10','decoder20'],scale_channels)})
        if et_head:
            from .et_head import ETResidualHead
            self.et_head=ETResidualHead()

    def forward(self,image,availability_mask,mode='infer',teacher_latent=None,task_target=None):
        if mode not in self.MODES:raise ValueError(mode)
        training_flow=mode in ('flow_train','joint_train')
        if teacher_latent is not None and not training_flow:
            raise ValueError('Teacher features are forbidden during inference/utility training')
        if task_target is not None and not training_flow:
            raise ValueError('Segmentation targets are forbidden outside flow/joint training')
        mask=validate_mask(availability_mask,image.shape[0]).to(image.device)
        encoded=self.backbone.encode(image,mask)
        observed=encoded['latent']
        if mode!='baseline' and not self.disable_rfm:encoded['logits']=encoded['logits'].detach()
        if self.disable_dmg:
            adjacency=observed.new_zeros(observed.shape[0],4,3)
            need=observed.new_zeros(observed.shape[0],3)
        else:
            adjacency,need=self.graph(observed,mask)
        fm=observed.sum()*0.
        if training_flow and not self.disable_rfm:
            if teacher_latent is None or task_target is None:
                raise ValueError('v21 flow training requires teacher features and training labels')
            task_error=self.flow.task_residual_target(encoded['logits'],task_target,observed.shape[2:])
            fm=self.flow.loss(observed,teacher_latent,adjacency,need,mask,task_error)
        if mode=='flow_train':return dict(loss_fm=fm,observed=observed,adjacency=adjacency,region_need=need)
        residual=torch.zeros_like(observed)
        if mode!='baseline' and not self.disable_rfm:
            # Full-modal samples deliberately use this same learned task path.
            residual=self.flow.sample(observed,adjacency,need,mask,self.nfe,
                method=self.solver,enable_grad=mode in ('joint_train','seg_train'),deterministic=True)
        utility,std,region_gate,_=self.utility(observed,residual,adjacency,mask)
        if mode=='baseline' or self.disable_rfm:
            logits=encoded['logits'];scales={}
        else:
            logits,scales=self.backbone.decode_refined(observed,encoded['context'],mask,
                residual,region_gate,self.refinement,
                gated=mode!='ungated' and not self.disable_cfg)
        if hasattr(self,'et_head') and mode!='baseline':
            logits=self.et_head(logits,image,mask)
        output=dict(logits=logits,baseline_logits=encoded['logits'].detach(),
            observed=observed,residual=residual,adjacency=adjacency,region_need=need,
            utility=utility,utility_std=std,region_gate=region_gate,loss_fm=fm,
            spatial_gates={k:v['region_gate'] for k,v in scales.items()},
            scale_residuals={k:v['residual'] for k,v in scales.items()})
        if mode in ('joint_train','utility_train'):
            with torch.no_grad():
                if self.disable_rfm:
                    output['ungated_logits']=encoded['logits'].detach()
                else:
                    output['ungated_logits'],_=self.backbone.decode_refined(observed,encoded['context'],mask,
                        residual.detach(),region_gate.detach(),self.refinement,gated=False)
        return output
