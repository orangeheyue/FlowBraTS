"""Queue ET-focused fine-tuning after the active v2.1 baseline run releases GPU 0."""
import json
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/finetune_brats2020/finetune/best.pt"
OUTPUT = ROOT / "outputs/finetune_brats2020_et"
STATE = OUTPUT / "queue.json"
ACTIVE_PID = 183522


def alive(pid):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists():
        return False
    return stat.read_text().split()[2] != "Z"


def write_state(**values):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    current = json.loads(STATE.read_text()) if STATE.exists() else {}
    current.update(values, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2) + "\n")
    temporary.replace(STATE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-now", action="store_true",
                        help="Start on GPU 0 immediately, even while baseline evaluation is active")
    args = parser.parse_args()
    write_state(supervisor_pid=os.getpid(),
                status="starting" if args.start_now else "waiting_for_v21_baseline",
                wait_pid=None if args.start_now else ACTIVE_PID)
    while not args.start_now and alive(ACTIVE_PID):
        time.sleep(30)
    if not SOURCE.is_file():
        write_state(status="failed", error=f"missing source checkpoint: {SOURCE}")
        raise FileNotFoundError(SOURCE)
    if (OUTPUT / "last.pt").exists():
        write_state(status="failed", error="output already contains last.pt")
        raise FileExistsError(OUTPUT / "last.pt")
    command = [sys.executable, "-u", "-m", "scripts.train",
               "--config", "configs/finetune_brats2020_et.yaml",
               "--output", str(OUTPUT), "--initialize", str(SOURCE)]
    with (OUTPUT / "console.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write_state(status="running", worker_pid=process.pid, command=command)
        returncode = process.wait()
    write_state(status="complete" if returncode == 0 else "failed", returncode=returncode)
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
