"""Measure real v2.1 end-to-end optimizer-step memory without saving weights."""
import argparse
import json
import time

import torch

from scripts.train import compute_loss, prepare_target, set_stage
from src.config import build_model, load_config, seed_everything
from src.data.m2f_data import build_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--batch-size', type=int, required=True)
    args = p.parse_args()
    config = load_config(args.config)
    seed_everything(config['seed'])
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    model = build_model(config).cuda().train()
    set_stage(model, 'end_to_end')
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6)
    ds = build_dataset(config['m2ftrans']['root'], config['data']['root'],
                       config['data']['split_dir'], 'train', config['data'].get('split_index'))
    image, target, mask, _ = next(iter(torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=0)))
    image = image.cuda()
    target = prepare_target(target.cuda())
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for pattern in (None, True, False):
        masks = mask.cuda()
        if pattern is True:
            masks[:] = True
        elif pattern is False:
            masks[:] = False
            masks[:, 1] = True
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, parts = compute_loss(model, image, target, masks, 'end_to_end')
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite probe loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        del loss, parts
    torch.cuda.synchronize()
    print(json.dumps(dict(batch_size=args.batch_size,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
        elapsed_seconds=time.monotonic()-start, load_report=model.backbone.load_report)), flush=True)


if __name__ == '__main__':
    main()
