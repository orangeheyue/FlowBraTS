"""Flow targets include segmentation error even for fully observed samples."""
import torch
from torch.nn import functional as F
from .flow import ConditionalResidualFlow


class TaskResidualFlow(ConditionalResidualFlow):
    def __init__(self,channels=256,hidden=64,flow_channels=64,task_scale=.25,
                 task_class_weights=(1.,1.,1.,1.),velocity_type='residual'):
        super().__init__(channels,hidden,flow_channels,velocity_type)
        generator=torch.Generator().manual_seed(21037)
        basis=torch.randn(flow_channels,4,generator=generator)
        basis=basis/basis.norm(dim=0,keepdim=True).clamp_min(1e-8)
        self.register_buffer('task_basis',basis)
        self.task_scale=task_scale
        if len(task_class_weights)!=4 or any(float(x)<=0 for x in task_class_weights):
            raise ValueError('task_class_weights must contain four positive values')
        self.task_class_weights=tuple(float(x) for x in task_class_weights)

    def task_residual_target(self, baseline_logits, target, spatial_size):
        with torch.no_grad():
            labels=target.long()
            labels=torch.where(labels==4,3,labels)
            if labels.min()<0 or labels.max()>3:raise ValueError('Invalid segmentation labels')
            truth=F.one_hot(labels,4).permute(0,4,1,2,3).float()
            error=truth-baseline_logits.detach().float().softmax(1)
            weights=torch.tensor(self.task_class_weights,device=error.device,dtype=error.dtype)
            error=error*weights[None,:,None,None,None]
            error=F.adaptive_avg_pool3d(error,spatial_size)
            task=torch.einsum('kc,bcdhw->bkdhw',self.task_basis.float(),error)
            return self.task_scale*task

    def loss(self,observed,teacher,adjacency,region_need,mask,task_error=None):
        if task_error is None:raise ValueError('v21 training requires a task residual target')
        observed_flow=self.to_flow(observed)
        with torch.no_grad():
            target=self.to_flow(teacher.detach())-observed_flow.detach()+task_error
        source=self.source(observed_flow,adjacency,region_need,mask)
        timestep=torch.rand(observed.shape[0],device=observed.device)
        sample=self.path.sample(x_0=source,x_1=target,t=timestep)
        prediction=self.velocity(sample.x_t,timestep,observed_flow,adjacency,region_need,mask)
        return (prediction.float()-sample.dx_t.float()).square().mean()
