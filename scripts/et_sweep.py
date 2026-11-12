"""Train ET variants on GPU 0, optionally in parallel; select on validation only."""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/finetune_brats2020/finetune/best.pt'
CONFIGS=['configs/et_sweep_a_sampling.yaml','configs/et_sweep_b_recall.yaml',
         'configs/et_sweep_c_finehead.yaml']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--parallel',action='store_true')
    parser.add_argument('--adopt-running',action='store_true',help='Keep existing workers from status.json')
    args=parser.parse_args()
    from src.config import load_config
    output=ROOT/('outputs/et_sweep_smoke' if args.smoke else 'outputs/et_sweep')
    output.mkdir(parents=True,exist_ok=True)
    lock=open(output/'supervisor.lock','a')
    import fcntl
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
    previous_records={r['name']:r for r in previous.get('experiments',[])}
    state=dict(supervisor_pid=os.getpid(),gpu=0,status='running',experiments=[],
               target_et_gain_pp=2.0,selection_split='val',smoke_only=args.smoke,
               execution='parallel' if args.parallel else 'sequential')
    def save():
        state['updated']=time.strftime('%Y-%m-%d %H:%M:%S')
        temporary=output/'status.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(output/'status.json')
    save()
    active=[]
    def is_alive(pid):
        path=Path(f'/proc/{pid}/stat')
        return path.exists() and path.read_text().split()[2]!='Z'
    def wait_active():
        while active:
            for entry in list(active):
                record,process=entry
                if process is None:
                    if is_alive(record['worker_pid']):continue
                    returncode=0 if (Path(record['output'])/'TRAIN_COMPLETE.json').exists() else 1
                else:
                    returncode=process.poll()
                    if returncode is None:continue
                record.update(status='complete' if returncode==0 else 'failed',returncode=returncode)
                active.remove(entry)
                save()
            if active:time.sleep(5)
    for cfg in CONFIGS:
        config=load_config(cfg)
        folder=output/Path(config['train']['output']).name
        folder.mkdir(parents=True,exist_ok=True)
        record=dict(name=config['experiment']['name'],config=cfg,output=str(folder))
        state['experiments'].append(record)
        prior=previous_records.get(record['name'],{})
        pid=prior.get('worker_pid')
        if pid and is_alive(pid):
            cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if not args.adopt_running or cfg.encode() not in cmdline or str(folder).encode() not in cmdline:
                raise RuntimeError(f'Refusing duplicate or mismatched worker: {pid}')
            record.update(status='running',worker_pid=pid,adopted=True,command=prior.get('command'))
            active.append((record,None))
            save()
            if not args.parallel:wait_active()
            continue
        if (folder/'TRAIN_COMPLETE.json').exists() and not args.smoke:
            record.update(status='complete',reused=True)
            save()
            continue
        command=[sys.executable,'-u','-m','scripts.train','--config',cfg,'--output',str(folder)]
        if args.smoke:
            command+=['--initialize',str(SOURCE),'--epochs','1','--smoke-batches','2']
        elif (folder/'last.pt').exists():
            command+=['--resume',str(folder/'last.pt')]
        else:
            command+=['--initialize',str(SOURCE)]
        with (folder/'console.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record.update(status='running',worker_pid=process.pid,command=command)
            save()
        active.append((record,process))
        if not args.parallel:wait_active()
    wait_active()
    if any(r['status']=='failed' for r in state['experiments']):
        state.update(status='failed')
        save()
        raise SystemExit(1)
    if args.smoke:
        state.update(status='smoke_complete')
        save()
        return
    import torch
    baseline=json.loads(Path(load_config(CONFIGS[0])['validation']['initial_summary']).read_text())
    base_et=baseline['mean_dice'][2]
    winner=None
    for record in state['experiments']:
        payload=torch.load(Path(record['output'])/'best.pt',map_location='cpu',weights_only=False)
        best=payload['best_validation']
        record.update(best_epoch=best['epoch'],validation_dice=best['mean_dice'],
            validation_hd95=best['mean_hd95'],et_gain_pp=100*(best['mean_dice'][2]-base_et))
        if best['epoch']>0 and best['mean_dice'][2]>base_et+1e-4 and (
                winner is None or best['mean_dice'][2]>winner['validation_dice'][2]):
            winner=record
        del payload
    state['winner']=winner['name'] if winner else 'original_v21_epoch20'
    state['target_achieved_on_validation']=bool(winner and winner['et_gain_pp']>=2.0)
    state['test_used_for_variant_selection']=False
    if winner is None:
        state.update(status='complete_no_validation_improvement')
        save()
        return
    state.update(status='testing_validation_winner')
    save()
    from src.config import build_model
    from src.checkpoint import load_checkpoint
    from src.evaluation import evaluate
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    config=load_config(winner['config'])
    model=build_model(config).cuda().eval()
    load_checkpoint(model,Path(winner['output'])/'best.pt')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    summary=evaluate(model,config,'test',Path(winner['output'])/'test_best')
    state.update(status='complete',winner_test_dice=summary['mean_dice'],
                 winner_test_hd95=summary['mean_hd95'])
    save()


if __name__=='__main__':
    main()
