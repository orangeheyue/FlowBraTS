"""Run the complete BraTS2020 sampler matrix on physical GPUs 0 and 1."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from scripts.sampler_ablation import MASKS, SETTINGS


GPU_QUEUES = {
    0: ('euler_s2', 'euler_s4', 'euler_s6', 'heun3_s2'),
    1: ('euler_s1', 'euler_s8', 'rk4_s2'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = {name: (solver, steps) for name, solver, steps in SETTINGS}
    lock = threading.Lock()
    state_path = output/'parallel_status.json'
    state = dict(status='running_validation', started_unix=time.time(),
                 physical_gpus=[0, 1], gpu_queues=GPU_QUEUES, completed=[])

    def save_state():
        state_path.write_text(json.dumps(state, indent=2) + '\n')

    def run_queue(gpu):
        log_path = output/f'gpu{gpu}.log'
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           PYTHONUNBUFFERED='1',
                           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        for name in GPU_QUEUES[gpu]:
            solver, steps = settings[name]
            command = [sys.executable, '-u', '-m', 'scripts.sampler_worker',
                       '--config', args.config, '--checkpoint', args.checkpoint,
                       '--output', str(output), '--name', name,
                       '--solver', solver, '--steps', str(steps)]
            with log_path.open('a') as stream:
                result = subprocess.run(command, env=environment,
                                        stdout=stream, stderr=subprocess.STDOUT,
                                        check=False)
            if result.returncode:
                raise RuntimeError(f'{name} failed on GPU{gpu}: {result.returncode}')
            with lock:
                state['completed'].append(name)
                state['updated_unix'] = time.time()
                save_state()

    save_state()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_queue, gpu) for gpu in (0, 1)]
            for future in futures:
                future.result()
    except Exception as error:
        state.update(status='failed', error=repr(error), updated_unix=time.time())
        save_state()
        raise

    records = []
    for name, _, _ in SETTINGS:
        records.append(json.loads((output/name/'record.json').read_text()))
    best = max(records, key=lambda item: (item['mean_wt_tc_et_dice'],
                                          -item['mean_wt_tc_et_hd95'],
                                          -item['velocity_nfe']))
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                  dataset='BraTS2020', selection_split='val',
                  validation_masks=MASKS, default_baseline='euler_s2',
                  execution='two physical GPUs; one serial queue per GPU',
                  settings=records, selected=best, test_used_for_selection=False)
    (output/'ablation_summary.json').write_text(json.dumps(report, indent=2) + '\n')

    state.update(status='running_selected_test', selected=best,
                 updated_unix=time.time())
    save_state()
    solver = 'heun' if best['solver'] == 'heun3' else best['solver']
    command = [sys.executable, '-u', '-m', 'scripts.sampler_worker',
               '--config', args.config, '--checkpoint', args.checkpoint,
               '--output', str(output), '--name', 'selected_test',
               '--solver', solver, '--steps', str(best['steps']), '--split', 'test']
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with (output/'selected_test.log').open('a') as stream:
        result = subprocess.run(command, env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        state.update(status='failed', returncode=result.returncode,
                     updated_unix=time.time())
        save_state()
        raise SystemExit(result.returncode)
    tested = json.loads((output/'selected_test'/'record.json').read_text())
    report.update(selected_test_summary=tested['summary'],
                  selected_test_mean_dice=tested['mean_dice'],
                  selected_test_mean_hd95=tested['mean_hd95'],
                  selected_test_wall_seconds=tested['wall_seconds'])
    (output/'ablation_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (output/'COMPLETE').write_text(
        'Dual-GPU validation and validation-selected 15-mask test completed.\n')
    state.update(status='complete', updated_unix=time.time())
    save_state()


if __name__ == '__main__':
    main()
