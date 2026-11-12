"""Local task residuals with spatially varying region/channel gates."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialResidualRefiner(nn.Module):
    def __init__(self, channels, latent_channels=256, hidden=32):
        super().__init__()
        self.feature=nn.Conv3d(channels,hidden,1)
        self.condition=nn.Conv3d(latent_channels,hidden,1)
        self.local=nn.Sequential(
            nn.Conv3d(hidden*2,hidden*2,3,padding=1,groups=hidden*2),
            nn.Conv3d(hidden*2,hidden,1),nn.GroupNorm(min(8,hidden),hidden),nn.SiLU())
        self.residual=nn.Conv3d(hidden,channels,1)
        self.spatial=nn.Conv3d(hidden,3,1)
        self.channel_region=nn.Parameter(torch.zeros(channels,3))
        # Small, nonzero residuals preserve a conservative warm start while
        # allowing gradients through the entire correction branch immediately.
        nn.init.normal_(self.residual.weight,std=1e-3)
        nn.init.zeros_(self.residual.bias)
        nn.init.normal_(self.spatial.weight,std=1e-3)
        nn.init.zeros_(self.spatial.bias)

    def forward(self, feature, flow_residual, region_gate, gated=True):
        condition=self.condition(flow_residual)
        condition=F.interpolate(condition,size=feature.shape[2:],mode='trilinear',align_corners=False)
        hidden=self.local(torch.cat((self.feature(feature),condition),1))
        residual=self.residual(hidden)
        prior=torch.logit(region_gate.float().clamp(1e-4,1-1e-4)).to(hidden.dtype)
        spatial_gate=torch.sigmoid(self.spatial(hidden)+prior[:,:,None,None,None])
        channel_gate=torch.einsum('brdhw,cr->bcdhw',spatial_gate,self.channel_region.softmax(1).to(spatial_gate.dtype))
        corrected=feature+(channel_gate*residual if gated else residual)
        return corrected,dict(residual=residual,region_gate=spatial_gate,channel_gate=channel_gate)
