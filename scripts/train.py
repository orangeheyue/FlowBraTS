import argparse
import json
import logging
import random
import time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.checkpoint import full_state, innovation_state, load_checkpoint, initialize_from_v1
from src.config import build_model, load_config, seed_everything
from src.data.m2f_data import build_dataset
from src.losses import segmentation_loss, safety_loss, utility_loss, utility_target
from src.evaluation import evaluate, better, baseline_cache


def set_stage(model, stage):
    """Desc: 按 specv1 的分阶段策略选择可训练模块。"""
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = {
        "flow": (model.graph, model.flow),
        "utility": (model.utility,),
        "joint": (model.graph, model.flow, model.utility),
        "end_to_end": (model,),
    }
    if hasattr(model,"refinement") and stage in ("joint","utility"):
        modules[stage]=(*modules[stage],model.refinement)
    if stage not in modules:
        raise ValueError("stage must be flow, utility, joint or end_to_end")
    for module in modules[stage]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def prepare_target(target):
    """Desc: 将 M2FTrans 训练集的 one-hot 标签还原为类别索引。"""
    if target.ndim == 5 and target.shape[1] == 4:
        return target.argmax(dim=1)
    return target.long()


def compute_loss(model, image, target, mask, stage, loss_config=None, objective=None):
    """Desc: 计算对应阶段的统一训练目标。"""
    teacher = model.extract_teacher(image) if stage in ("flow", "joint", "end_to_end") else None
    mode = {
        "flow": "flow_train", "utility": "utility_train",
        "joint": "joint_train", "end_to_end": "joint_train",
    }[stage]
    extra={"task_target":target} if getattr(model,"version",None)=="v21" and teacher is not None else {}
    output = model(image, mask, mode=mode, teacher_latent=teacher,**extra)
    objective=objective or {}
    fm_weight=float(objective.get('fm_weight',1.0))
    utility_weight=float(objective.get('utility_weight',.2))
    safety_weight=float(objective.get('safety_weight',.2))
    sparse_weight=float(objective.get('graph_sparse_weight',.01))
    if stage == "flow":
        loss = fm_weight*output["loss_fm"] + sparse_weight*model.graph.sparse_loss(output["adjacency"])
        return loss, {"fm": output["loss_fm"].detach()}

    paired_target = utility_target(
        output["baseline_logits"], output["ungated_logits"], target
    )
    loss_utility = utility_loss(output["utility"], output["utility_std"], paired_target)
    loss_seg = segmentation_loss(output["logits"], target, **(loss_config or {}))
    loss_safe = safety_loss(output["logits"], output["baseline_logits"], target)
    loss = loss_seg + utility_weight*loss_utility + safety_weight*loss_safe
    if stage in ("joint", "end_to_end"):
        loss = loss + fm_weight*output["loss_fm"]
    if stage == "end_to_end":
        loss = loss + sparse_weight*model.graph.sparse_loss(output["adjacency"])
    return loss, {
        "seg": loss_seg.detach(), "utility": loss_utility.detach(),
        "safe": loss_safe.detach(), "fm": output["loss_fm"].detach(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=("flow", "utility", "joint", "end_to_end"))
    parser.add_argument("--resume")
    parser.add_argument("--initialize",help="Transfer v1 weights into v21; start a new optimizer/epoch count")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accumulation", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    stage = args.stage or config["train"]["stage"]
    epochs = args.epochs or config["train"]["epochs"]
    config["train"].update(stage=stage,epochs=epochs,
        batch_size=args.batch_size or config["train"]["batch_size"],
        grad_accumulation=args.grad_accumulation or config["train"]["grad_accumulation"])
    seed_everything(config["seed"])
    torch.set_num_threads(config["train"].get("cpu_threads",4))
    if config.get("fast",False):
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic=False
        torch.set_float32_matmul_precision("high")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("M2FTrans_v1 contains CUDA-bound attention masks; training requires CUDA")
    device_name = config["device"]
    device = torch.device("cuda:0" if device_name == "cuda" else device_name)
    torch.cuda.set_device(device.index)

    model = build_model(config).to(device)
    # Architecture construction consumes different amounts of random numbers.
    # Reset the training stream so all velocity ablations see identical sampling.
    seed_everything(config["seed"])
    if args.resume and args.initialize:raise ValueError("Choose either --resume or --initialize")
    transfer=initialize_from_v1(model,args.initialize) if args.initialize else None
    resumed = load_checkpoint(model,args.resume) if args.resume else None
    if resumed and resumed.get("smoke_only") and not args.smoke_batches:
        raise ValueError("Cannot start a formal run from a smoke checkpoint")
    set_stage(model, stage)
    if config['train'].get('scope')=='decoder':
        for name,p in model.backbone.named_parameters():
            p.requires_grad=name.startswith('backbone.decoder_fusion.')
    logging.info("Backbone initialization: %s",model.backbone.load_report)
    logging.info("Trainable parameters: backbone=%d, additions=%d",sum(p.numel() for p in model.backbone.parameters() if p.requires_grad),sum(p.numel() for name,p in model.named_parameters() if not name.startswith('backbone.') and p.requires_grad))
    if stage == "end_to_end":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(),
                 "lr": config["train"]["backbone_lr"]},
                {"params": [p for name,p in model.named_parameters() if not name.startswith("backbone.")],
                 "lr": config["train"]["lr"]},
            ],
            weight_decay=config["train"]["weight_decay"],
        )
    else:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=config["train"]["lr"],
            weight_decay=config["train"]["weight_decay"]
        )
    amp_dtype=torch.bfloat16 if config.get("amp_dtype")=="bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and amp_dtype==torch.float16)
    start=0;best=None;bad_checks=0;last_validation_epoch=0
    if resumed and resumed.get("stage")==stage:
        optimizer.load_state_dict(resumed["optimizer"])
        start=resumed["epoch"]
        if resumed.get("scaler") and scaler.is_enabled():scaler.load_state_dict(resumed["scaler"])
        best=resumed.get("best_validation");bad_checks=resumed.get("bad_checks",0)
        last_validation_epoch=resumed.get("last_validation_epoch",0)
        rng=resumed.get("rng")
        if rng:
            random.setstate(rng["python"]);np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"]);torch.cuda.set_rng_state_all(rng["cuda"])
    logging.info("Start epoch=%d, total=%d, batch=%d, accumulation=%d, AMP=%s",start,epochs,config["train"]["batch_size"],config["train"]["grad_accumulation"],amp_dtype)
    dataset = build_dataset(
        config["m2ftrans"]["root"], config["data"]["root"],
        config["data"]["split_dir"], "train",config["data"].get("split_index"),
    )
    if config['data'].get('et_sampling'):
        from src.data.et_dataset import ETPatchDataset
        dataset=ETPatchDataset(config['data']['root'],config['data']['split_dir'],
            config['data'].get('split_index'),config['data'].get('patch_size',80),
            **config['data']['et_sampling'])
    loader = DataLoader(
        dataset, batch_size=args.batch_size or config["train"]["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config['seed']),
        num_workers=config["train"]["workers"], pin_memory=True,
        persistent_workers=config["train"]["workers"]>0,
        **({"prefetch_factor":2} if config["train"]["workers"]>0 else {}),
    )
    output_dir = Path(args.output or config["train"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulation = args.grad_accumulation or config["train"]["grad_accumulation"]
    if accumulation < 1:
        raise ValueError("grad accumulation must be positive")

    if transfer:(output_dir/"initialization.json").write_text(json.dumps(transfer,indent=2)+"\n")
    (output_dir/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    if (output_dir/"last.pt").exists() and not args.resume:
        raise FileExistsError("Existing output checkpoint; use --resume or a new output")
    validation=config.get("validation",{})
    if resumed:
        previous_validation=resumed.get("config",{}).get("validation",{})
        if (previous_validation.get("masks") != validation.get("masks")
                or previous_validation.get("selection_config") != validation.get("selection_config")):
            logging.info("Validation protocol changed; resetting selection and revalidating resumed epoch %d",start)
            best=None;bad_checks=0;last_validation_epoch=0
    def save(name,epoch):
        payload = {
            "epoch":epoch,"stage":stage,
            "model":full_state(model) if stage=="end_to_end" else innovation_state(model),
            "checkpoint_type":"full" if stage=="end_to_end" else "innovations",
            "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"config":config,
            "smoke_only":bool(args.smoke_batches),"best_validation":best,"bad_checks":bad_checks,
            "last_validation_epoch":last_validation_epoch,
            "rng":{"python":random.getstate(),"numpy":np.random.get_state(),
                   "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}}
        temp=output_dir/(name+".tmp");torch.save(payload,temp);temp.replace(output_dir/name)
    def validate_epoch(completed_epoch):
        nonlocal best,bad_checks,last_validation_epoch
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        result=evaluate(model,config,"val",output_dir/f"val_epoch_{completed_epoch:04d}",mask_names=validation.get("masks"))
        improved=better(result,best,selection=validation.get("selection_config"))
        if improved:
            best=dict(result,epoch=completed_epoch);bad_checks=0
        else:bad_checks+=1
        last_validation_epoch=completed_epoch
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
        if improved:save("best.pt",completed_epoch)
        save("last.pt",completed_epoch)
        print(json.dumps({"validation_epoch":completed_epoch,"improved":improved,
            "best_epoch":best["epoch"],"mean_dice":result["mean_dice"],
            "mean_hd95":result["mean_hd95"],"bad_checks":bad_checks}),flush=True)
        return validation.get("patience",0)>0 and bad_checks>=validation["patience"]
    if transfer and validation.get('initial_summary') and not args.smoke_batches:
        if model.flow.task_class_weights != (1.,1.,1.,1.):
            raise ValueError('Cached initial summary requires unchanged inference parameters')
        initial=json.loads(Path(validation['initial_summary']).read_text())
        if initial['split']!='val' or [c['mask'] for c in initial['combinations']]!=validation['masks']:
            raise ValueError('Initial summary must match validation masks and split')
        best=dict(initial,epoch=0)
        save('best.pt',0)
        logging.info('Initial checkpoint participates in selection: ET Dice %.6f',best['mean_dice'][2])
    early_stopped=bool(best is not None and validation.get("patience",0)>0 and bad_checks>=validation["patience"])
    if (not args.smoke_batches and validation.get("every") and start>0
            and (start%validation["every"]==0 or start==epochs) and last_validation_epoch<start):
        early_stopped=validate_epoch(start)
    for epoch in range(start,epochs):
        if early_stopped:break
        epoch_start=time.monotonic()
        # Separate learning rates preserve the pretrained backbone while allowing
        # new components to adapt; use a slow cosine decay over remaining epochs.
        schedule_start=config["train"].get("schedule_start_epoch",start)
        fraction=(epoch-schedule_start)/max(1,epochs-schedule_start)
        ratio=.1+.9*.5*(1+np.cos(np.pi*fraction))
        for group,base in zip(optimizer.param_groups,
                [config["train"]["backbone_lr"],config["train"]["lr"]] if stage=="end_to_end" else [config["train"]["lr"]]):
            group["lr"]=base*ratio
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        if stage != "end_to_end":
            model.backbone.eval()
        elif config['train'].get('scope')=='decoder':
            model.backbone.eval()
            model.backbone.backbone.decoder_fusion.train()
        optimizer.zero_grad(set_to_none=True)
        running = {}
        for step, (image, target, mask, _name) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            target = prepare_target(target.to(device, non_blocking=True))
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=config["amp"],dtype=amp_dtype):
                if config['train'].get('supervised_only'):
                    output=model(image,mask,mode='seg_train')
                    loss=segmentation_loss(output['logits'],target,**config.get('loss',{}))
                    parts={'seg':loss.detach()}
                    del output
                else:
                    loss, parts = compute_loss(model, image, target, mask, stage,
                                               config.get("loss"),config.get("objective"))
                # Normalize the final short accumulation group correctly.
                remaining=len(loader)%accumulation
                group_size=remaining if remaining and step>=len(loader)-remaining else accumulation
                loss = loss / group_size
            if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
            scaler.scale(loss).backward()
            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config["train"].get("max_grad_norm",1.0),error_if_nonfinite=not scaler.is_enabled())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + float(value)
            if config['data'].get('et_sampling'):
                positive=(target==3).flatten(1).any(1).float().mean()
                running['et_positive_patch_fraction']=running.get('et_positive_patch_fraction',0.)+float(positive)
            if step%10==0:
                print(json.dumps({"epoch":epoch+1,"step":step+1,"batches":len(loader),
                    "loss":float(loss.detach())*group_size,"seconds":round(time.monotonic()-epoch_start,2)}),flush=True)
            if args.smoke_batches and step + 1 >= args.smoke_batches:
                break

        batches = step + 1
        if batches % accumulation:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config["train"].get("max_grad_norm",1.0),error_if_nonfinite=not scaler.is_enabled())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        record = {"epoch": epoch + 1, "stage": stage,"epoch_seconds":round(time.monotonic()-epoch_start,2),
                  **{key: value / batches for key, value in running.items()},
                  "peak_gpu_allocated_mib": round(
                      torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1
                  ),
                  "peak_gpu_reserved_mib": round(
                      torch.cuda.max_memory_reserved(device) / 1024 ** 2, 1
                  )}
        print(json.dumps(record, ensure_ascii=False))
        save("last.pt",epoch+1)
        if not args.smoke_batches and validation.get("every") and ((epoch+1)%validation["every"]==0 or epoch+1==epochs):
            # Release training graphs before full-volume validation.
            del loss,parts,image,target,mask
            torch.cuda.empty_cache()
            if validate_epoch(epoch+1):
                logging.info("Early stopped at epoch %d",epoch+1);break
    if args.smoke_batches:return
    (output_dir/'TRAIN_COMPLETE.json').write_text(json.dumps({
        'best_epoch':best['epoch'] if best else None,'bad_checks':bad_checks,
        'selection_split':'val','smoke_only':False},indent=2)+'\n')
    if validation.get("auto_test"):
        if not (output_dir/"best.pt").exists():raise RuntimeError("No validation-selected best checkpoint")
        load_checkpoint(model,output_dir/"best.pt")
        generated=evaluate(model,config,"test",output_dir/"test_best")
        # Restore the original published backbone for the paired comparison;
        # the official baseline wrapper bypasses all Flow/graph/gate modules.
        baseline_checkpoint=config.get("comparison",{}).get("checkpoint",config["m2ftrans"]["checkpoint"])
        if baseline_checkpoint is None:
            (output_dir/"comparison.json").write_text(json.dumps({
                "status":"dataset_specific_baseline_checkpoint_required",
                "dataset":config["data"].get("year"),"best_epoch":best["epoch"],
                "flowbrats_mean_dice":generated["mean_dice"],
                "flowbrats_mean_hd95":generated["mean_hd95"],
                "claimed_superiority":False,"test_used_for_tuning":False},indent=2)+"\n")
            (output_dir/"COMPLETE").write_text("Training, validation selection and all 15 test combinations completed; matched M2FTrans baseline unavailable.\n")
            return
        baseline=baseline_cache(config,generated)
        if baseline is None:
            model.backbone._load_strict(baseline_checkpoint)
            baseline=evaluate(model,config,"test",output_dir/"test_m2ftrans",mode="official_baseline")
        else:
            logging.info("Reused matching complete M2FTrans baseline: %s",baseline['reused_from'])
            cache_out=output_dir/"test_m2ftrans";cache_out.mkdir(exist_ok=True)
            (cache_out/"summary.json").write_text(json.dumps(baseline,indent=2)+"\n")
        gd=np.asarray([r["dice"] for r in generated["combinations"]])
        bd=np.asarray([r["dice"] for r in baseline["combinations"]])
        comparison={"best_epoch":best["epoch"],"flowbrats_mean_dice":generated["mean_dice"],
            "m2ftrans_mean_dice":baseline["mean_dice"],"delta_dice":(gd-bd).mean(0).tolist(),
            "per_mask_delta_dice":(gd-bd).tolist(),
            "all_15_masks_mean_WT_TC_ET_improved":bool(((gd[:,:3]-bd[:,:3]).mean(1)>0).all()),
            "overall_mean_WT_TC_ET_improved":bool(gd[:,:3].mean()>bd[:,:3].mean()),
            "flowbrats_mean_hd95":generated["mean_hd95"],"m2ftrans_mean_hd95":baseline["mean_hd95"],
            "selection_split":"val","selection_masks":validation.get("masks","all_15"),
            "baseline_reused_from":baseline.get("reused_from"),"test_used_for_tuning":False}
        (output_dir/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
        (output_dir/"COMPLETE").write_text("Training, validation selection and paired 15-mask test completed.\n")


if __name__ == "__main__":
    main()
