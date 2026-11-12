"""Run all matched CAM experiments on GPU1, resume cache, then write figures/CSV."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import MODEL_LABELS
from cam_report import report


def main():
    p=argparse.ArgumentParser(description=__doc__);here=Path(__file__).resolve().parent
    p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--years',type=int,nargs='+',default=[2018,2020],choices=[2018,2020])
    p.add_argument('--render-only',action='store_true')
    a=p.parse_args();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    if not a.render_only:
        for year in a.years:
            for model in MODEL_LABELS:
                subprocess.run([sys.executable,'-u',str(here/'cam_worker.py'),
                                '--year',str(year),'--model',model],env=env,check=True,cwd=here)
    report(here/'selection.json',a.years,here/'cam_cache',here/'cam_outputs')


if __name__=='__main__':main()
