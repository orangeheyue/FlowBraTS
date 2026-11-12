"""Run one sampler setting on one CUDA-visible GPU."""
import argparse
import json
import time
from pathlib import Path

from src.checkpoint import load_checkpoint
from src.config import build_model, load_config, seed_everything
from src.evaluation import evaluate
from scripts.sampler_ablation import MASKS, make_record, profile_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--solver', choices=('euler', 'heun', 'rk4'), required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    args = parser.parse_args()

    config = load_config(args.config)
    if 'BraTS2020' not in config['data']['root']:
        raise ValueError('This experiment must use BraTS2020')
    seed_everything(config['seed'])
    model = build_model(config).cuda().eval()
    payload = load_checkpoint(model, args.checkpoint)
    if payload.get('smoke_only'):
        raise RuntimeError('Smoke checkpoint is forbidden')
    model.solver, model.nfe = args.solver, args.steps
    folder = Path(args.output)/args.name
    begin = time.monotonic()
    result = evaluate(model, config, args.split, folder,
                      mask_names=MASKS if args.split == 'val' else None)
    elapsed = time.monotonic() - begin
    if args.split == 'val':
        profile_result = profile_flops(model, config, args.solver, args.steps)
        record = make_record(args.name, args.solver, args.steps, result,
                             folder/'summary.json', profile_result,
                             wall_seconds=elapsed)
    else:
        record = dict(name=args.name, solver=args.solver, steps=args.steps,
                      split='test', wall_seconds=elapsed,
                      summary=str((folder/'summary.json').resolve()),
                      mean_dice=result['mean_dice'], mean_hd95=result['mean_hd95'])
    (folder/'record.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
