"""Fine ET/NCR split correction preserving WT and TC probabilities exactly."""
import torch
from torch import nn
from torch.nn import functional as F
from src.data.protocol import apply_mask


class ETResidualHead(nn.Module):
    def __init__(self, hidden=12):
        super().__init__()
        self.features = nn.Sequential(nn.Conv3d(8,hidden,3,padding=1),
            nn.GroupNorm(3,hidden),nn.SiLU(),
            nn.Conv3d(hidden,hidden,3,padding=1,groups=hidden),nn.SiLU())
        self.offset = nn.Conv3d(hidden,1,1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, logits, image, mask):
        inputs = torch.cat((apply_mask(image,mask),logits.softmax(1)),1)
        inputs = F.avg_pool3d(inputs,2)
        delta = self.offset(self.features(inputs))
        delta = F.interpolate(delta,size=logits.shape[2:],mode='trilinear',align_corners=False)[:,0]
        pair = torch.logsumexp(logits[:,[1,3]].float(),dim=1)
        split = logits[:,3].float()-logits[:,1].float()+delta.float()
        # exp(l1)+exp(l3) is unchanged: all WT/TC probabilities are preserved.
        return torch.stack((logits[:,0].float(),pair+F.logsigmoid(-split),
                            logits[:,2].float(),pair+F.logsigmoid(split)),1)
