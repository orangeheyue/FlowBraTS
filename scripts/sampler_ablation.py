"""Complete BraTS2020 sampler ablation on one fixed checkpoint.

All weights and inputs are fixed. Only the ODE solver/integration step count is
changed. Profiler FLOPs describe one deterministic 80^3 missing-T1CE patch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.checkpoint import load_checkpoint
from src.config import build_model, load_config, seed_everything
from src.evaluation import evaluate


BASELINE = ('euler_s2', 'euler', 2)
SETTINGS = (
    ('euler_s1', 'euler', 1),
    ('euler_s2', 'euler', 2),
    ('euler_s4', 'euler', 4),
    ('euler_s6', 'euler', 6),
    ('euler_s8', 'euler', 8),
    ('heun3_s2', 'heun', 2),
    ('rk4_s2', 'rk4', 2),
)
MASKS = ['flairt1cet1t2', 't1cet1t2', 'flairt1t2', 'flair', 't1c']
CALLS_PER_STEP = {'euler': 1, 'heun': 3, 'rk4': 4}


def score(result):
    return (float(np.mean(result['mean_dice'][:3])),
            float(np.mean(result['mean_hd95'][:3])))


def load_summary(path, split, expected_masks=None):
    path = Path(path)
    result = json.loads(path.read_text())
    masks = [item['mask'] for item in result.get('combinations', [])]
    if result.get('split') != split:
        raise ValueError(f'{path} is not a {split} summary')
    if expected_masks is not None and masks != expected_masks:
        raise ValueError(f'{path} does not match the required masks')
    if not np.isfinite(result['mean_dice'] + result['mean_hd95']).all():
        raise ValueError(f'{path} contains non-finite metrics')
    return result


def profile_flops(model, config, solver, steps):
    """Return parameter count and observed FLOPs for one model forward."""
    from torch.profiler import ProfilerActivity, profile

    model.solver, model.nfe = solver, steps
    patch = int(config['data']['patch_size'])
    image = torch.zeros((1, 4, patch, patch, patch), device='cuda')
    mask = torch.tensor([[True, False, True, True]], device='cuda')
    with torch.inference_mode():
        model(image, mask)
    torch.cuda.synchronize()
    with torch.inference_mode(), profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True) as profiler:
        model(image, mask)
    torch.cuda.synchronize()
    flops = int(sum(event.flops or 0 for event in profiler.key_averages()))
    params = sum(parameter.numel() for parameter in model.parameters())
    del image, mask
    torch.cuda.empty_cache()
    return dict(params=params, params_million=params / 1e6,
                profiled_flops=flops, profiled_gflops=flops / 1e9,
                profile_patch=[1, 4, patch, patch, patch],
                profile_mask='flairt1t2',
                flops_method='torch.profiler observed FLOPs')


def make_record(name, solver, steps, result, summary, profile_result,
                wall_seconds=None, reused_metrics=False):
    dice, hd95 = score(result)
    return dict(name=name,
                solver='heun3' if solver == 'heun' else solver,
                steps=steps,
                velocity_nfe=steps * CALLS_PER_STEP[solver],
                mean_wt_tc_et_dice=dice,
                mean_wt_tc_et_hd95=hd95,
                wall_seconds=wall_seconds,
                summary=str(Path(summary).resolve()),
                reused_metrics=reused_metrics,
                **profile_result)


def write_report(path, checkpoint, records, selected=None, **extra):
    report = dict(checkpoint=str(Path(checkpoint).resolve()), dataset='BraTS2020',
                  selection_split='val', validation_masks=MASKS,
                  default_baseline='euler_s2', settings=records,
                  test_used_for_selection=False, **extra)
    if selected is not None:
        report['selected'] = selected
    path.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-selected-test', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output/'ablation_summary.json'
    config = load_config(args.config)
    if 'BraTS2020' not in config['data']['root']:
        raise ValueError('This experiment must use BraTS2020')
    seed_everything(config['seed'])
    model = build_model(config).cuda().eval()
    payload = load_checkpoint(model, args.checkpoint)
    if payload.get('smoke_only'):
        raise RuntimeError('Smoke checkpoint is forbidden')

    records = []
    write_report(summary_path, args.checkpoint, records)

    for name, solver, steps in SETTINGS:
        seed_everything(config['seed'])
        model.solver, model.nfe = solver, steps
        begin = time.monotonic()
        result = evaluate(model, config, 'val', output/name, mask_names=MASKS)
        elapsed = time.monotonic() - begin
        profile_result = profile_flops(model, config, solver, steps)
        records.append(make_record(name, solver, steps, result,
                                   output/name/'summary.json', profile_result,
                                   wall_seconds=elapsed))
        write_report(summary_path, args.checkpoint, records)

    best = max(records, key=lambda item: (item['mean_wt_tc_et_dice'],
                                          -item['mean_wt_tc_et_hd95'],
                                          -item['velocity_nfe']))
    report = write_report(summary_path, args.checkpoint, records, selected=best)
    if args.run_selected_test:
        model.solver = 'heun' if best['solver'] == 'heun3' else best['solver']
        model.nfe = best['steps']
        seed_everything(config['seed'])
        tested = evaluate(model, config, 'test', output/'selected_test')
        report['selected_test_summary'] = str(
            (output/'selected_test/summary.json').resolve())
        report['selected_test_reused'] = False
        report['selected_test_mean_dice'] = tested['mean_dice']
        report['selected_test_mean_hd95'] = tested['mean_hd95']
        summary_path.write_text(json.dumps(report, indent=2) + '\n')
        (output/'COMPLETE').write_text(
            'Validation-selected sampler and one 15-mask test completed.\n')


if __name__ == '__main__':
    main()
