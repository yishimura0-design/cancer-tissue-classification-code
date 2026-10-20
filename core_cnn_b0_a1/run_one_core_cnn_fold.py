#!/usr/bin/env python3
"""Core-independent outer CV CNN training with separate INNER early-stopping set.

Run one scenario x architecture x outer fold. Does not change existing originals.
Rare classes can be absent from inner validation but MUST remain in training.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dl_grouped_common import (
    CancerImageDataset, build_model, set_backbone_frozen,
    moderate_class_weights, prepare_dataframe, evaluate, seed_everything
)
from core_eval import aggregate_core_predictions, core_log_loss, core_metrics
from run_contract import contract, require_new_or_matching


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--roles',required=True,help='inner_outer_roles.csv from make_core_splits.py')
    p.add_argument('--scenario',choices=['scenario1','scenario2'],required=True)
    p.add_argument('--model',choices=['resnet18','efficientnet_b0','mobilenet_v3_large'],required=True)
    p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--freeze-epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--label-smoothing',type=float,default=0.05)
    p.add_argument('--grad-clip',type=float,default=1.0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--save-checkpoint',action='store_true')
    p.add_argument('--experiment-id',default='B0',help='B0 baseline or A1 variant')
    a=p.parse_args()
    if a.epochs <= 0 or a.patience <= 0:
        p.error('--epochs and --patience must be positive')
    return a


def loader(df,train,a):
    return DataLoader(CancerImageDataset(df,train,train,a.image_size),
                      batch_size=a.batch_size,shuffle=train,
                      num_workers=a.workers,pin_memory=torch.cuda.is_available(),
                      persistent_workers=a.workers>0)


def to_core(group_preds, data_df, class_names):
    mapping=data_df[['group_id','sample_id']].drop_duplicates().rename(columns={'sample_id':'core_id'})
    return aggregate_core_predictions(group_preds,mapping,class_names)


def load_roles(path,fold):
    roles=pd.read_csv(path)
    required={'outer_fold','core_id','role','scenario2_label'}
    if not required.issubset(roles.columns):
        raise ValueError(f'roles missing {required-set(roles)}')
    part=roles[roles.outer_fold==fold].copy()
    if len(part)!=45 or part.core_id.nunique()!=45:
        raise ValueError(f'Expected each fold to partition exactly 45 cores (found {len(part)})')
    if not {'train','val','test'}.issubset(set(part.role)) or not set(part.role).issubset({'train','val','test'}):
        raise ValueError('Train, validation, test must all be nonempty')
    return part


def main():
    a=parse_args();seed_everything(a.seed)
    out=Path(a.out)/a.scenario/a.model/f'fold_{a.fold}'
    here=Path(__file__).resolve().parent
    identity=contract({'manifest':a.manifest,'roles':a.roles},
        [here/x for x in ['run_one_core_cnn_fold.py','dl_grouped_common.py','core_eval.py','run_contract.py']],
        {'scenario':a.scenario,'model':a.model,'fold':a.fold,'experiment_id':a.experiment_id,
         **{k:v for k,v in vars(a).items() if k not in {'manifest','roles','out','scenario','model','fold','experiment_id'}}})
    if require_new_or_matching(out,identity,['test_image_predictions.csv.gz','test_core_predictions.csv']):
        print(f'Verified completed fold (same inputs/config/code): {out}');return
    df,classes=prepare_dataframe(a.manifest,a.scenario)   # original transforms & labels
    roles=load_roles(a.roles,a.fold)
    merged=df.merge(roles[['core_id','role']],left_on='sample_id',right_on='core_id',how='left',validate='many_to_one')
    if merged.role.isna().any():
        raise RuntimeError('Missing core role for an input image')
    if not (merged.role == 'val').any():
        raise ValueError('Inner validation is required; never train fixed epochs without it')
    split={role:merged[merged.role==role].reset_index(drop=True) for role in ['train','val','test']}
    for r in ['train','val','test']:
        print(f'{r}: {len(split[r])} images, {split[r].sample_id.nunique()} cores',flush=True)
    ids={r:set(split[r].sample_id) for r in split}
    if any(ids[a] & ids[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise RuntimeError('Core-level overlap across roles')
    if len(ids['train']|ids['val']|ids['test']) != 45:
        raise RuntimeError('Core split does not cover all 45 cores')
    if set(split['train'].label)!=set(classes):
        raise RuntimeError('Training lacks class(es)')
    if split['train'].empty or split['test'].empty or split['val'].empty:
        raise RuntimeError('Train, inner validation and outer test MUST each contain cores')
    if set(split['val'].label)& {'Stage III','Stage IV'}:
        raise RuntimeError('Sparse classes unexpectedly placed in inner validation')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp=bool(a.amp and dev.type=='cuda')
    print(f'CUDA={amp} device={dev}',flush=True)
    tl=loader(split['train'],True,a)
    vl=loader(split['val'],False,a)
    model=build_model(a.model,len(classes),True).to(dev)
    set_backbone_frozen(model,True)
    crit=nn.CrossEntropyLoss(weight=moderate_class_weights(split['train'],len(classes)).to(dev),
                             label_smoothing=a.label_smoothing,reduction='none')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=a.head_lr,weight_decay=a.weight_decay)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[];best_loss=float('inf');best_ep=0;left=a.patience;best_state=None
    for ep in range(1,a.epochs+1):
        if ep==a.freeze_epochs+1:
            set_backbone_frozen(model,False)
            opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
            sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=2)
        model.train();total=seen=0
        for batch in tl:
            x=batch['image'].to(dev,non_blocking=True)
            y=batch['target'].to(dev,non_blocking=True)
            w=batch['sample_weight'].to(dev).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,enabled=amp):
                loss=(crit(model(x),y)*w).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
            scaler.step(opt);scaler.update()
            total+=float(loss.item())*len(y);seen+=len(y)
        row={'epoch':ep,'train_loss':total/max(seen,1),'lr':opt.param_groups[0]['lr']}
        _,_,field_predictions=evaluate(model,vl,dev,classes,amp)
        val_cores=to_core(field_predictions,split['val'],classes)
        val_nll=core_log_loss(val_cores,classes)
        row['val_core_nll']=val_nll
        sch.step(val_nll)
        if val_nll<best_loss-1e-7:
            best_loss=val_nll;best_ep=ep;left=a.patience
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            left-=1
        history.append(row)
        print(row,flush=True)
        if left<=0:
            break
    if best_state is None:
        raise RuntimeError('No best validation checkpoint selected')
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(out/'epoch_history.csv',index=False)
    if a.save_checkpoint:
        torch.save(model.state_dict(),out/'best_model.pt')
    # Core-independent outer test: called ONCE, only after epoch choice.
    test_loader=loader(split['test'],False,a)
    field_metrics, image_pred, field_pred=evaluate(model,test_loader,dev,classes,amp)
    core_pred=to_core(field_pred,split['test'],classes)
    core_pred.insert(0,'fold',a.fold)
    core_pred.insert(0,'experiment_id',a.experiment_id)
    core_pred.insert(1,'scenario',a.scenario)
    core_pred.insert(2,'model',a.model)
    # One IMAGE prediction per held-out angle; retain FOV and physical Core identity.
    lookup=split['test'][['path','sample_id','group_id']].drop_duplicates('path')
    if lookup.path.duplicated().any():
        raise RuntimeError('Duplicate image paths in outer Test')
    image_pred=image_pred.merge(lookup[['path','sample_id']],on='path',how='left',validate='one_to_one')
    if image_pred.sample_id.isna().any() or len(image_pred)!=len(split['test']):
        raise RuntimeError('IMAGE Test prediction coverage mismatch')
    image_pred=image_pred.rename(columns={'sample_id':'core_id','group_id':'fov_id'})
    image_pred.insert(0,'fold',a.fold)
    image_pred.insert(0,'experiment_id',a.experiment_id)
    image_pred.insert(1,'scenario',a.scenario)
    image_pred.insert(2,'model',a.model)
    field_pred=field_pred.rename(columns={'group_id':'fov_id'})
    image_pred.to_csv(out/'test_image_predictions.csv.gz',index=False,compression='gzip')
    field_pred.to_csv(out/'test_field_predictions.csv',index=False)
    core_pred.to_csv(out/'test_core_predictions.csv',index=False)
    result={
      'scenario':a.scenario,'model':a.model,'fold':a.fold,'experiment_id':a.experiment_id,
      'n_train_cores':len(ids['train']),'n_inner_val_cores':len(ids['val']),
      'n_outer_test_cores':len(ids['test']),'best_epoch':best_ep,
      'best_val_core_nll':best_loss,
      'validation_rare_class_support':{
          label:int(split['val'].loc[split['val'].label==label,'sample_id'].nunique()) for label in ['Stage III','Stage IV']},
      'early_stopping':f'inner_core_NLL_patience_{a.patience}',
      'outer_test_used_for_checkpoint_selection':False,
      'outer_test_core_metrics':core_metrics(core_pred,classes),
      'outer_test_image_count':int(len(image_pred)),
      'outer_test_core_id_set':sorted(ids['test']),
      'run_contract':identity,
      'outer_test_field_metrics':field_metrics,
      'classes':classes,
      'seed':a.seed,'epochs_ran':len(history),
      'note':'Only core-independent, not proven patient-independent. Outer fold may miss some classes.',
    }
    (out/'fold_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
