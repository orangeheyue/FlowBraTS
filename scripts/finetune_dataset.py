"""Run the v2.1 pretrained warmup/finetune workflow for one dataset."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=(2018, 2021), required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    out = ROOT / f'outputs/finetune_brats{args.year}'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(supervisor_pid=os.getpid(), gpu=args.gpu, year=args.year,
                     initialization='dataset-specific original M2FTrans', stages=[])
        def save():
            temp = out / 'process.json.tmp'
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(out / 'process.json')
        save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, TZ='Asia/Shanghai',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        for phase in ('warmup', 'finetune'):
            folder = out / phase
            folder.mkdir(exist_ok=True)
            marker = folder / 'PHASE_COMPLETE'
            if marker.exists():
                state['stages'].append(dict(phase=phase, status='complete'))
                save()
                continue
            suffix = '_warmup' if phase == 'warmup' else ''
            config = f'configs/finetune_brats{args.year}{suffix}.yaml'
            command = [sys.executable, '-u', '-m', 'scripts.train', '--config', config,
                       '--output', str(folder)]
            if (folder / 'last.pt').exists():
                command += ['--resume', str(folder / 'last.pt')]
            elif phase == 'finetune':
                command += ['--resume', str(out / 'warmup/last.pt')]
            with (folder / 'console.log').open('a') as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
            record = dict(phase=phase, pid=proc.pid, command=command, status='running')
            state['stages'].append(record)
            save()
            print(f'BraTS{args.year} {phase}: PID {proc.pid}', flush=True)
            code = proc.wait()
            record.update(status='complete' if code == 0 else 'failed', returncode=code)
            save()
            if code:
                state['status'] = 'failed'
                save()
                raise SystemExit(code)
            marker.write_text('Phase completed.\n')
        state['status'] = 'complete'
        save()


if __name__ == '__main__':
    main()
