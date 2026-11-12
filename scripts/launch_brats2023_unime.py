"""Start the GPU-0 experiment in an independent session and record its PID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/BraTS2023_UniME'
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / 'process.json'
    if (output / 'last.pt').exists() and not args.resume:
        raise RuntimeError('Formal checkpoint exists; use the documented resume command.')
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Previous process {previous['pid']} is still alive.")
    command = ['bash', str(root / 'scripts/train_brats2023_unime_gpu1.sh')]
    if args.resume:
        if not (output/'last.pt').is_file():raise FileNotFoundError(output/'last.pt')
        command += ['--resume',str(output/'last.pt')]
    with (output / 'console.log').open('a') as log:
        process = subprocess.Popen(
            command, cwd=root, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    record = dict(
        pid=process.pid, launched_at_utc=datetime.now(timezone.utc).isoformat(),
        command=command, physical_gpu=0, seed=42, batch_size=8,
        grad_accumulation=2, output=str(output),
    )
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
