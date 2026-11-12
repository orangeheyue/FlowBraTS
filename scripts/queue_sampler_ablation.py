"""Wait for GPU 1 and completed velocity runs, then execute sampler ablations."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ('mlp', 'transformer3d', 'mamba3d', 'unet3d')
OUTPUT = ROOT/'outputs/sampler_ablation_brats2020'
STATE = OUTPUT/'queue_state.json'


def write(**values):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    current = json.loads(STATE.read_text()) if STATE.exists() else {}
    current.update(values, updated_unix=time.time())
    STATE.write_text(json.dumps(current, indent=2)+'\n')


def gpu_state():
    raw = subprocess.check_output(['nvidia-smi', '-i', '1',
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    memory, utilization = (int(value.strip()) for value in raw.split(','))
    return memory, utilization


def velocity_processes_alive():
    result = subprocess.run(['pgrep', '-f', 'scripts.train --config configs/velocity_'],
                            check=False, capture_output=True, text=True)
    return bool(result.stdout.strip())


def choose_velocity():
    candidates = []
    for name in RUNS:
        folder = ROOT/'outputs/velocity_ablation'/name
        completion = json.loads((folder/'TRAIN_COMPLETE.json').read_text())
        epoch = int(completion['best_epoch'])
        summary_path = folder/f'val_epoch_{epoch:04d}'/'summary.json'
        result = json.loads(summary_path.read_text())
        dice = float(np.mean(result['mean_dice'][:3]))
        hd95 = float(np.mean(result['mean_hd95'][:3]))
        candidates.append(dict(name=name, best_epoch=epoch, dice=dice, hd95=hd95,
                               config=str(ROOT/f'configs/velocity_{name}_brats2020.yaml'),
                               checkpoint=str(folder/'best.pt'), summary=str(summary_path)))
    selected = max(candidates, key=lambda item: (item['dice'], -item['hd95']))
    (OUTPUT/'velocity_selection.json').write_text(json.dumps(dict(
        selection_split='val', candidates=candidates, selected=selected,
        test_used_for_selection=False), indent=2)+'\n')
    return selected


def main():
    write(status='waiting_velocity_runs', supervisor_pid=os.getpid(), physical_gpu=1)
    while True:
        complete = all((ROOT/'outputs/velocity_ablation'/name/'TRAIN_COMPLETE.json').exists()
                       for name in RUNS)
        if complete and not velocity_processes_alive():
            break
        time.sleep(60)
    selected = choose_velocity()
    write(status='waiting_gpu_idle', selected_velocity=selected)
    idle_checks = 0
    while idle_checks < 3:
        memory, utilization = gpu_state()
        idle_checks = idle_checks + 1 if memory <= 10000 and utilization <= 20 else 0
        write(status='waiting_gpu_idle', selected_velocity=selected,
              gpu_memory_mib=memory, gpu_utilization=utilization,
              consecutive_idle_checks=idle_checks)
        time.sleep(60)
    write(status='running', selected_velocity=selected)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [sys.executable, '-u', '-m', 'scripts.sampler_ablation',
               '--config', selected['config'], '--checkpoint', selected['checkpoint'],
               '--output', str(OUTPUT), '--run-selected-test']
    with (OUTPUT/'console.log').open('a') as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    write(status=('complete' if result.returncode == 0 else 'failed'),
          returncode=result.returncode)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
