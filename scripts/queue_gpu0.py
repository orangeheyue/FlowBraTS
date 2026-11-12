"""Attach to BraTS2020 and run independent 2018/2021 experiments on GPU 0."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'outputs/gpu0_queue.json'


def alive(pid,expected):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        return expected in cmd and Path(f'/proc/{pid}/cwd').resolve()==ROOT
    except (FileNotFoundError,ProcessLookupError,PermissionError):
        return False


def main():
    if STATE.exists():
        previous=json.loads(STATE.read_text())
        if alive(previous['supervisor_pid'],'scripts.queue_gpu0'):
            raise RuntimeError('GPU 0 queue is already running')
    attached=json.loads((ROOT/'outputs/gpu0_brats2020_e2e_300/process.json').read_text())
    state=dict(supervisor_pid=os.getpid(),physical_gpu=0,jobs=[
        dict(year=2020,pid=attached['pid'],status='attached',output=str(ROOT/'outputs/gpu0_brats2020_e2e_300')),
        dict(year=2018,status='queued',output=str(ROOT/'outputs/gpu0_brats2018_e2e_300')),
        dict(year=2021,status='queued',output=str(ROOT/'outputs/gpu0_brats2021_e2e_300'))])
    def save():
        tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
    save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',TZ='Asia/Shanghai',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
    for job in state['jobs']:
        out=Path(job['output'])
        if (out/'COMPLETE').exists():job['status']='complete';save();continue
        if job['year']==2020:
            while alive(job['pid'],'scripts.train'):time.sleep(20)
            job['status']='complete' if (out/'COMPLETE').exists() else 'failed'
            save();continue
        out.mkdir(parents=True,exist_ok=True)
        config=f'configs/gpu0_brats{job["year"]}_e2e.yaml'
        command=[sys.executable,'-u','-m','scripts.train','--config',config,'--output',str(out)]
        if (out/'last.pt').exists():command+=['--resume',str(out/'last.pt')]
        with (out/'console.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        job.update(status='running',pid=proc.pid,command=command);save()
        (out/'process.json').write_text(json.dumps(job,indent=2)+'\n')
        print(f'Started BraTS{job["year"]}: PID {proc.pid}',flush=True)
        code=proc.wait()
        job.update(status='complete' if code==0 and (out/'COMPLETE').exists() else 'failed',returncode=code)
        save()
    state['finished']=True;save()


if __name__=='__main__':main()
