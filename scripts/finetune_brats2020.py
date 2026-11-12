"""Run M2FTrans-initialized frozen warmup then low-LR v21 finetuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='0')
    args=parser.parse_args()
    out=ROOT/'outputs/finetune_brats2020';out.mkdir(parents=True,exist_ok=True)
    state_path=out/'process.json'
    if state_path.exists():
        previous=json.loads(state_path.read_text())
        pid=previous.get('supervisor_pid')
        if pid and pid!=os.getpid():
            try:
                command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            except FileNotFoundError:command=''
            if 'scripts.finetune_brats2020' in command:raise RuntimeError('Finetuning supervisor already running')
    state=dict(supervisor_pid=os.getpid(),gpu=args.gpu,initialization='original M2FTrans BraTS2020 pretrained checkpoint',stages=[])
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,TZ='Asia/Shanghai',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    for phase,config in [('warmup','configs/finetune_brats2020_warmup.yaml'),('finetune','configs/finetune_brats2020.yaml')]:
        folder=out/phase;folder.mkdir(exist_ok=True)
        marker=folder/'PHASE_COMPLETE'
        if marker.exists():
            state['stages'].append(dict(phase=phase,status='complete',output=str(folder)));save();continue
        command=[sys.executable,'-u','-m','scripts.train','--config',config,'--output',str(folder)]
        if (folder/'last.pt').exists():command+=['--resume',str(folder/'last.pt')]
        elif phase=='finetune':command+=['--resume',str(out/'warmup/last.pt')]
        with (folder/'console.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(phase=phase,pid=proc.pid,command=command,status='running',output=str(folder))
        state['stages'].append(record);save()
        print(f'Started {phase}: PID {proc.pid}',flush=True)
        code=proc.wait()
        record.update(status='complete' if code==0 else 'failed',returncode=code);save()
        if code:
            state['status']='failed';save();raise SystemExit(code)
        marker.write_text('Phase completed.\n')
    state['status']='complete';save()


if __name__=='__main__':main()
