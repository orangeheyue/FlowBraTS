"""Isolated decoder-stage hooks; no changes to upstream M2FTrans source."""
from .m2f_adapter import M2FTransAdapter


class M2FTransV21Adapter(M2FTransAdapter):
    def decode_refined(self, latent, context, mask, flow_residual, region_gate, refinement, gated=True):
        records={}
        latent,records['bottleneck']=refinement['bottleneck'](latent,flow_residual,region_gate,gated)
        decoder=self.backbone.decoder_fusion
        handles=[]
        def hook_for(name):
            def hook(_module,_inputs,output):
                result,records[name]=refinement[name](output,flow_residual,region_gate,gated)
                return result
            return hook
        try:
            for name,module in [('decoder10',decoder.d4_out),('decoder20',decoder.d3_out)]:
                handles.append(module.register_forward_hook(hook_for(name)))
            logits=self.decode(latent,context,mask)
        finally:
            for handle in handles:handle.remove()
        return logits,records
