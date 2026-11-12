"""GT-conditioned, final-decoder Seg-Grad-CAM for the four saved models."""
import argparse
import json
from pathlib import Path
import types
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import (CHECKPOINTS, SCENARIOS, atomic_json, load_case,
                    seed_everything, stable_seed, window_starts)
from infer_worker import load_network, ProbabilityAdapter

CLASSES = ((2, 'ED'), (1, 'NCR_NET'), (3, 'ET'))


def cache_path(root, year, model, case, scenario):
    return Path(root) / f'BraTS{year}' / model / case / f'{scenario}.npz'


class DecoderCAM:
    """Keep exact forward values, expose gradients of final decoder features only.

    A leaf at each classifier input avoids retaining encoder/sampler graphs. This
    is attribution to the final decoder conditional on the realised sampler path.
    DiffUNet combines the edge head and every accumulated DDIM head occurrence.
    """
    def __init__(self, model, kind):
        self.model, self.kind = model, kind
        self.features, self.handles = [], []
        model.requires_grad_(False)
        if kind == 'rfnet':
            model.decoder_fuse.softmax = nn.Identity()
            heads = [('decoder_fuse.seg_layer', model.decoder_fuse.seg_layer)]
        elif kind == 'm2ftrans':
            model.decoder_fusion.softmax = nn.Identity()
            heads = [('decoder_fusion.seg_layer', model.decoder_fusion.seg_layer)]
        elif kind == 'flowbrats':
            head = model.backbone.backbone.decoder_fusion.seg_layer
            heads = [('backbone.backbone.decoder_fusion.seg_layer', head)]
        else:
            # Original DDIM is decorated no_grad. Unwrap on this loaded instance
            # only; argmax feedback still defines the original discrete trajectory.
            sampler = model.denoise_model.ddim_sample
            model.denoise_model.ddim_sample = types.MethodType(sampler.__wrapped__, model.denoise_model)
            heads = []
            for name in ('edge_model', 'denoise_model'):
                decoder = getattr(model, name).decoder
                # seg_layers[-1] is the full-resolution classifier in nnU-Net.
                heads.append((name + '.decoder.seg_layers.-1', decoder.seg_layers[-1]))
        self.layer_names = [name + ':input' for name, _ in heads]
        for _, head in heads:
            self.handles.append(head.register_forward_pre_hook(self.capture))

    def capture(self, _module, inputs):
        feature = inputs[0].detach().requires_grad_(True)
        self.features.append(feature)
        return (feature, *inputs[1:])

    def forward(self, image, mask):
        self.features.clear()
        if self.kind in ('rfnet', 'm2ftrans'):
            return self.model(image, mask)
        if self.kind == 'flowbrats':
            return self.model(image, mask, mode='infer')['logits']
        image = image * mask[:, :, None, None, None]
        shape = image.shape[2:]
        pads = [(-d) % 32 for d in shape]
        image = F.pad(image, (0, pads[2], 0, pads[1], 0, pads[0]))
        return self.model(image, ddim=True)[:, :, :shape[0], :shape[1], :shape[2]]

    def plane(self, volume, seg, target_z, available, device):
        height, width, depth, _ = volume.shape
        heat = np.zeros((3, height, width), np.float64)
        probabilities = np.zeros((4, height, width), np.float64)
        weights = np.zeros((height, width), np.float64)
        starts = [(h, w, z) for h in window_starts(height)
                  for w in window_starts(width) for z in window_starts(depth)
                  if z <= target_z < z + 80]
        coverage = np.zeros((height, width), np.float32)
        for h, w, _ in starts:
            coverage[h:h+80, w:w+80] += 1
        plane_gt = np.asarray(seg[:, :, target_z])
        totals = [int((plane_gt == label).sum()) for label, _ in CLASSES]
        mask = torch.tensor([available], device=device, dtype=torch.bool)
        for h, w, z in starts:
            patch = np.array(volume[h:h+80, w:w+80, z:z+80], dtype=np.float32)
            image = torch.from_numpy(patch.transpose(3, 0, 1, 2)[None]).to(device)
            logits = self.forward(image, mask)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite logits')
            local_z = target_z - z
            probabilities[:, h:h+80, w:w+80] += logits.detach().softmax(1)[0, :, :, :, local_z].cpu().numpy()
            weights[h:h+80, w:w+80] += 1
            gt = plane_gt[h:h+80, w:w+80]
            for index, (label, _) in enumerate(CLASSES):
                region = (gt == label).astype(np.float32)
                if not region.any():
                    continue
                # Each target voxel has equal influence after accounting for
                # reference-window overlap. GT sets the explanation ROI only.
                roi = region / coverage[h:h+80, w:w+80] / totals[index]
                roi = torch.from_numpy(roi).to(device)
                score = (logits[0, label, :, :, local_z] * roi).sum()
                grads = torch.autograd.grad(score, self.features, retain_graph=True, allow_unused=True)
                maps = []
                for feature, gradient in zip(self.features, grads):
                    if gradient is None:
                        continue
                    alpha = gradient.mean(dim=(2, 3, 4), keepdim=True)
                    cam = (alpha * feature).sum(dim=1, keepdim=True)
                    if cam.shape[2:] != image.shape[2:]:
                        # DiffUNet is padded to 96 before its full-res head.
                        cam = cam[:, :, :80, :80, :80]
                    maps.append(cam.detach()[0, 0, :, :, local_z])
                if not maps:
                    raise RuntimeError('No gradient-connected classifier features')
                cam = torch.stack(maps).sum(0).relu()
                if not torch.isfinite(cam).all():
                    raise FloatingPointError('Nonfinite CAM')
                heat[index, h:h+80, w:w+80] += cam.cpu().numpy()
            del logits, image
            self.features.clear()
        return (heat / weights[None]).astype(np.float32), (probabilities / weights[None]).argmax(0).astype(np.uint8)


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--year', type=int, required=True, choices=[2018, 2020])
    p.add_argument('--model', required=True, choices=list(CHECKPOINTS[2018]))
    p.add_argument('--manifest', type=Path, default=here/'selection.json')
    p.add_argument('--cache-root', type=Path, default=here/'cam_cache')
    p.add_argument('--case-limit', type=int)
    p.add_argument('--scenario-limit', type=int)
    p.add_argument('--verify-forward', action='store_true', help='Check one real patch against original inference and exit')
    args = p.parse_args()
    torch.set_num_threads(4)
    seed_everything(1037)
    cases = json.loads(args.manifest.read_text())['datasets'][str(args.year)]['cases'][:args.case_limit]
    model, info = load_network(args.model, args.year, CHECKPOINTS[args.year][args.model])
    model = model.cuda().eval()
    if args.verify_forward:
        volume, _ = load_case(args.year, cases[0]['case_id'])
        patch = np.array(volume[:80,:80,:80],dtype=np.float32)
        image = torch.from_numpy(patch.transpose(3,0,1,2)[None]).cuda()
        mask = torch.ones((1,4),dtype=torch.bool,device='cuda')
        seed_everything(1037)
        with torch.no_grad():
            reference = ProbabilityAdapter(model,args.model)(image,mask).detach()
        extractor = DecoderCAM(model,args.model)
        seed_everything(1037)
        logits = extractor.forward(image,mask)
        actual = logits.softmax(1)
        error = float((actual-reference).detach().abs().max())
        assert torch.allclose(actual,reference,atol=1e-6,rtol=1e-5), error
        gradients = torch.autograd.grad(logits[:,3].mean(),extractor.features,allow_unused=True)
        assert any(g is not None and torch.isfinite(g).all() and g.abs().max()>0 for g in gradients)
        print(f'FORWARD + GRADIENT CHECK PASSED {args.model}: max probability difference={error}',flush=True)
        return
    extractor = DecoderCAM(model, args.model)
    for case in cases:
        volume, seg = load_case(args.year, case['case_id'])
        for scenario, _, available in SCENARIOS[:args.scenario_limit]:
            path = cache_path(args.cache_root, args.year, args.model, case['case_id'], scenario)
            if path.exists():
                with np.load(path) as cached:
                    if cached['cam'].shape != (3, *volume.shape[:2]):
                        raise ValueError(f'Invalid cache {path}')
                print(f'reuse {path}', flush=True)
                continue
            # Common random numbers across missing-modality settings.
            seed = stable_seed(args.year, case['case_id'], 'cam_common_noise', args.model)
            seed_everything(seed)
            before = time.time()
            cam, pred = extractor.plane(volume, seg, case['slice'], available, 'cuda:0')
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, cam=cam, prediction=pred, slice=case['slice'], seed=seed)
            tmp.replace(path)
            print(f'done {args.year} {args.model} {case["case_id"]} {scenario} '
                  f'{time.time()-before:.1f}s max={cam.max():.3g}', flush=True)
    atomic_json(args.cache_root/f'BraTS{args.year}'/args.model/'metadata.json',
                dict(checkpoint=str(CHECKPOINTS[args.year][args.model]), checkpoint_info=info,
                     layers=extractor.layer_names, classes=CLASSES,
                     target='GT class ROI mean axial pre-softmax logit; overlap compensated',
                     method='3D Seg-Grad-CAM, final decoder classifier inputs; FP32',
                     scope='conditional final-decoder attribution, not upstream sampler attribution',
                     common_seed_across_scenarios=True,
                     case_limit=args.case_limit, scenario_limit=args.scenario_limit))


if __name__ == '__main__':
    main()
