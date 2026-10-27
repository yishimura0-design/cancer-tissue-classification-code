#!/usr/bin/env python3
"""Leakage-controlled angle-CNN 5-fold CV, independently held-out by subgroup.

Unlike the historical 02_stage1_dl_5fold_search.py, OUTER test groups are
NEVER used in epoch selection, LR scheduler decisions, or early stopping.
Each OUTER training partition is divided into inner Train and Validation
groups. Every group is tested exactly once. No model is selected from test.

Actions:
  prepare   Freeze and audit the 5-fold 3-role assignments. CPU only.
  run       Train ONE architecture and ONE outer fold (independent PBS job).
  summarize Aggregate only if all 15 model/fold jobs are complete.

This file must live next to the original dl_common.py; no changes to
Classical ML or historical Angle CNN results are required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

MODELS = ('resnet18', 'efficientnet_b0', 'mobilenet_v3_large')
LABELS = ('HPNE', 'PANC', 'MIA')
FOLD_COUNT = 5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temp, path)


def build_role_table(df, seed=42, val_fraction=0.2, folds=FOLD_COUNT):
    """Balance folds by unique subgroup (NOT by number of angle images)."""
    required = {'group_key', 'target', 'image_path'}
    if not required.issubset(df):
        raise ValueError(f'Missing fields: {required-set(df.columns)}')
    if df.empty or df.image_path.duplicated().any():
        raise ValueError('Empty manifest or duplicated image paths')
    if df.group_key.isna().any() or df.group_key.astype(str).str.strip().eq('').any():
        raise ValueError('Missing subgroup identifier')
    group_sizes = df.groupby('group_key').size()
    group_labels = df.groupby('group_key')['target'].agg(['first', 'nunique'])
    if (group_labels['nunique'] != 1).any():
        raise ValueError('A subgroup contains multiple class labels')
    groups = pd.DataFrame({'group_key': group_labels.index.astype(str),
                           'target': group_labels['first'].astype(int).to_numpy(),
                           'n_images': group_sizes.to_numpy()}).sort_values('group_key').reset_index(drop=True)
    if set(groups.target) != set(range(len(LABELS))):
        raise ValueError('Expected all HPNE/PANC/MIA classes')
    if groups.target.value_counts().min() < folds:
        raise ValueError('Not enough independent subgroups per class for stratified outer CV')
    if not 0 < val_fraction < 0.5:
        raise ValueError('Validation fraction must be positive and < 0.5')

    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    records = []
    for fold, (outside, test_idx) in enumerate(outer.split(groups.group_key, groups.target), 1):
        outside_groups = groups.iloc[outside].reset_index(drop=True)
        inner = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                       random_state=seed + 1009 * fold)
        tr, va = next(inner.split(outside_groups.group_key, outside_groups.target))
        role_lookup = {g: 'train' for g in outside_groups.iloc[tr].group_key}
        role_lookup.update({g: 'val' for g in outside_groups.iloc[va].group_key})
        role_lookup.update({g: 'test' for g in groups.iloc[test_idx].group_key})
        if len(role_lookup) != len(groups):
            raise AssertionError('Missing or repeated outer/inner group assignment')
        for item in groups.itertuples(index=False):
            records.append({'fold': fold, 'group_key': item.group_key,
                            'target': item.target, 'n_images': item.n_images,
                            'role': role_lookup[item.group_key]})
    roles = pd.DataFrame.from_records(records)
    verify_role_table(df, roles, folds=folds)
    return roles


def verify_role_table(df, roles, folds=FOLD_COUNT):
    expected_groups = set(df.group_key.astype(str))
    if set(roles.fold) != set(range(1, folds + 1)):
        raise ValueError('Expected exactly 5 outer folds')
    outer_tests = []
    per_fold = []
    for f, part in roles.groupby('fold'):
        if len(part) != len(expected_groups) or part.group_key.duplicated().any():
            raise ValueError(f'Fold {f} has repeated or missing subgroup rows')
        if set(part.group_key) != expected_groups:
            raise ValueError(f'Fold {f} does not cover all subgroups')
        buckets = {role: set(part.loc[part.role == role, 'group_key'])
                   for role in ('train', 'val', 'test')}
        if any(not buckets[r] for r in buckets):
            raise ValueError(f'Empty Train / Validation / Test fold {f}')
        if any(buckets[a] & buckets[b] for a,b in [('train','val'),('train','test'),('val','test')]):
            raise ValueError(f'Subgroup overlap in fold {f}')
        if set(part.role) != {'train', 'val', 'test'}:
            raise ValueError(f'Unexpected role in fold {f}')
        for role in buckets:
            if set(part.loc[part.role == role, 'target']) != set(range(len(LABELS))):
                raise ValueError(f'Fold {f} / {role} lacks a class')
        for _, group in df.groupby('group_key'):
            if len(set(part.loc[part.group_key == str(group.group_key.iloc[0]), 'role'])) != 1:
                raise ValueError('Angular images from a subgroup have multiple roles')
        outer_tests.extend(buckets['test'])
        per_fold.append({'fold': int(f), **{f'{r}_groups': len(g) for r,g in buckets.items()},
                         **{f'{r}_images': int(part.loc[part.role == r, 'n_images'].sum()) for r in buckets}})
    if len(outer_tests) != len(expected_groups) or set(outer_tests) != expected_groups:
        raise ValueError('Outer test coverage must contain every subgroup exactly once')
    return per_fold


def check_manifest_metadata(df, expected_groups=37, expected_images=939):
    if len(df) != expected_images or df.group_key.nunique() != expected_groups:
        raise ValueError(f'Dataset changed: found {len(df)} images / {df.group_key.nunique()} subgroups, '
                         f'expected {expected_images} / {expected_groups}. Do not silently alter historical dataset.')
    if df.label.value_counts().to_dict() != {'HPNE':315,'PANC':321,'MIA':303}:
        raise ValueError('Class counts do not match audited Angle Sweep dataset')


def run_ident(a, manifest_hash, role_hash):
    return {'manifest_sha256': manifest_hash, 'roles_sha256': role_hash,
            'trainer_sha256': sha256(__file__),
            'shared_code_sha256': sha256(Path(__file__).with_name('dl_common.py')),
            'model': a.model, 'fold': a.fold,
            'config': {k:getattr(a,k) for k in ('seed','image_size','batch_size',
                         'head_epochs','max_epochs','min_epochs','early_stop_patience',
                         'head_lr','finetune_head_lr','backbone_lr','weight_decay',
                         'dropout','label_smoothing')}}


def prepare(a):
    from dl_common import load_manifest
    df = load_manifest(a.manifest)
    check_manifest_metadata(df)
    out = Path(a.out)
    if (out/'role_assignments.csv').exists() or (out/'run_metadata.json').exists():
        raise FileExistsError('A frozen CV plan already exists. Use a NEW --out path, do not overwrite it.')
    roles = build_role_table(df, seed=a.seed, val_fraction=a.inner_val_fraction)
    out.mkdir(parents=True, exist_ok=True)
    roles.to_csv(out/'role_assignments.csv', index=False)
    summary = verify_role_table(df, roles)
    metadata = {'design': 'outer_stratified_subgroup_5fold_inner_val_heldout',
                'group_definition': 'original manifest group_key; all angles stay together',
                'manifest': str(Path(a.manifest).resolve()),
                'manifest_sha256': sha256(a.manifest),
                'roles_sha256': sha256(out/'role_assignments.csv'),
                'n_groups': int(df.group_key.nunique()), 'n_images': len(df),
                'folds': summary, 'seed': a.seed,
                'inner_validation_fraction': a.inner_val_fraction,
                'caveat': 'Filename-derived subgroups; biological independence across distinct keys not proven'}
    json_write(out/'run_metadata.json', metadata)
    print(json.dumps({'status':'PASS','out':str(out),'folds':summary},indent=2))


def load_frozen_roles(a, df):
    out = Path(a.out)
    meta_path = out/'run_metadata.json'; csv_path = out/'role_assignments.csv'
    if not csv_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError('Run prepare action first; frozen roles are mandatory')
    meta = json.loads(meta_path.read_text())
    if sha256(a.manifest) != meta['manifest_sha256']:
        raise ValueError('Manifest hash differs from frozen plan')
    if sha256(csv_path) != meta['roles_sha256']:
        raise ValueError('Role assignment file changed')
    if a.seed != meta['seed'] or a.inner_val_fraction != meta['inner_validation_fraction']:
        raise ValueError('Split configuration differs from frozen plan')
    roles = pd.read_csv(csv_path)
    verify_role_table(df, roles)
    expected = build_role_table(df, seed=a.seed, val_fraction=a.inner_val_fraction)
    if not roles.equals(expected):
        raise ValueError('Frozen roles no longer match deterministic split generator')
    return roles, meta


def dataloader(df, training, args, fold):
    import torch
    from torch.utils.data import DataLoader
    from dl_common import CancerImageDataset, train_transform, eval_transform, subgroup_balanced_sampler
    ds = CancerImageDataset(df, train_transform(args.image_size) if training else eval_transform(args.image_size))
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
              pin_memory=torch.cuda.is_available(), persistent_workers=args.num_workers > 0)
    if training:
        kw['sampler'] = subgroup_balanced_sampler(df, args.seed+fold)
    else:
        kw['shuffle'] = False
    return DataLoader(ds, **kw)


def evaluate(model, loader, device, return_predictions=False):
    import torch
    from torch.cuda.amp import autocast
    from dl_common import compute_metrics
    model.eval()
    ys, ps, gs, paths = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device, non_blocking=True)
            with autocast(enabled=device.type=='cuda'):
                scores = model(x)
            ys.extend(batch['target'].tolist())
            ps.append(torch.softmax(scores, dim=1).cpu().numpy())
            gs.extend(list(batch['group_key']))
            paths.extend(list(batch['image_path']))
    if not ps:
        raise ValueError('Empty evaluation data')
    p = np.concatenate(ps)
    metrics = compute_metrics(ys, p, gs)
    if not return_predictions:
        return metrics
    img = pd.DataFrame({'image_path':paths,'group_key':gs,
                        'true_index':ys,'pred_index':p.argmax(axis=1)})
    for n,label in enumerate(LABELS):
        img[f'prob_{label}'] = p[:,n]
    unique_group_truth = img.groupby('group_key').true_index.nunique()
    if (unique_group_truth!=1).any():
        raise ValueError('Evaluation group contains mixed labels')
    means = img.groupby('group_key',sort=True)[[f'prob_{x}' for x in LABELS]].mean().reset_index()
    means['true_index'] = means.group_key.map(img.groupby('group_key').true_index.first().to_dict())
    means['pred_index'] = means[[f'prob_{x}' for x in LABELS]].to_numpy().argmax(axis=1)
    return metrics, img, means


def train_one(a):
    import torch
    import torch.nn as nn
    from torch.cuda.amp import GradScaler, autocast
    from dl_common import (load_manifest, build_model, set_seed, freeze_backbone,
                           split_parameter_groups, checkpoint_payload, metrics_to_dict)
    if a.model not in MODELS or not 1 <= a.fold <= FOLD_COUNT:
        raise ValueError('Require one recognized CNN architecture and a fold in 1..5')
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise RuntimeError('CUDA unavailable; do not schedule production training without a GPU')
    set_seed(a.seed + a.fold)
    df = load_manifest(a.manifest)
    check_manifest_metadata(df)
    roles, meta = load_frozen_roles(a, df)
    role_map = roles.loc[roles.fold == a.fold].set_index('group_key').role.to_dict()
    role_df = df.assign(role=df.group_key.map(role_map))
    if role_df.role.isna().any():
        raise AssertionError('An input angle image has no split role')
    train = role_df[role_df.role == 'train'].reset_index(drop=True)
    val = role_df[role_df.role == 'val'].reset_index(drop=True)
    test = role_df[role_df.role == 'test'].reset_index(drop=True)
    # Each of the three roles is populated, and whole subgroups never cross boundaries.
    group_sets = [set(x.group_key) for x in (train,val,test)]
    if any(group_sets[i]&group_sets[j] for i,j in ((0,1),(0,2),(1,2))):
        raise RuntimeError('FATAL: subgroup leakage')
    save_dir = Path(a.out)/'models'/a.model/f'fold_{a.fold}'
    if save_dir.exists():
        raise FileExistsError(f'Fold output exists: {save_dir}. Never merge with old/partial run.')
    save_dir.mkdir(parents=True)
    identity = run_ident(a, meta['manifest_sha256'], meta['roles_sha256'])
    json_write(save_dir/'identity.json', identity)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(a.model,3,a.dropout,True).to(device)
    tr_loader = dataloader(train, True, a, a.fold)
    val_loader = dataloader(val, False, a, a.fold)
    criterion = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    scaler = GradScaler(enabled=device.type == 'cuda')
    freeze_backbone(model,a.model,True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=a.head_lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.2,patience=3,min_lr=1e-7)
    best = float('-inf'); best_epoch = 0; no_improvement = 0; history=[]
    checkpoint = save_dir/'best.pt'
    for epoch in range(1,a.max_epochs+1):
        if epoch==a.head_epochs+1:
            freeze_backbone(model,a.model,False)
            body,head = split_parameter_groups(model,a.model)
            optimizer=torch.optim.AdamW([{'params':body,'lr':a.backbone_lr},
                                          {'params':head,'lr':a.finetune_head_lr}],weight_decay=a.weight_decay)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.2,patience=3,min_lr=1e-7)
        model.train(); total=0; count=0
        for batch in tr_loader:
            x=batch['image'].to(device,non_blocking=True)
            y=batch['target'].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type=='cuda'):
                loss=criterion(model(x),y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.item())*len(x);count+=len(x)
        # Inner validation is the ONLY data consulted for schedule, checkpoint and stopping.
        inner_metrics = evaluate(model,val_loader,device)
        scheduler.step(inner_metrics.group_macro_f1)
        history.append({'epoch':epoch,'train_loss':total/max(count,1),
                        **metrics_to_dict(inner_metrics,'inner_val_')})
        pd.DataFrame(history).to_csv(save_dir/'epoch_history.csv',index=False)
        if inner_metrics.group_macro_f1 > best + 1e-6:
            best=inner_metrics.group_macro_f1;best_epoch=epoch;no_improvement=0
            payload=checkpoint_payload(model,a.model,epoch,best,identity)
            torch.save(payload,checkpoint)
        else:
            no_improvement+=1
        print(f'{a.model} Fold {a.fold} Epoch {epoch}: inner_val_group_f1={inner_metrics.group_macro_f1:.4f}',flush=True)
        if epoch>=a.min_epochs and no_improvement>=a.early_stop_patience:
            break
    if not best_epoch:
        raise RuntimeError('No inner-validation checkpoint selected')
    # Test loader deliberately did not exist during training; ONLY now create it.
    try:
        payload=torch.load(checkpoint,map_location=device,weights_only=True)
    except TypeError:
        payload=torch.load(checkpoint,map_location=device)
    if payload.get('config') != identity:
        raise RuntimeError('Checkpoint identity mismatch')
    model.load_state_dict(payload['state_dict'])
    test_loader = dataloader(test, False, a, a.fold)
    test_metrics, image_predictions, group_predictions = evaluate(model,test_loader,device,return_predictions=True)
    image_predictions.insert(0,'fold',a.fold)
    group_predictions.insert(0,'fold',a.fold)
    image_predictions.to_csv(save_dir/'test_image_predictions.csv.gz',index=False,compression='gzip')
    group_predictions.to_csv(save_dir/'test_group_predictions.csv',index=False)
    result={'model':a.model,'fold':a.fold,'best_epoch':best_epoch,
            'early_stopping_metric':'inner_validation_subgroup_macro_f1',
            'outer_test_used_for_checkpoint_selection':False,
            'train_groups':len(group_sets[0]),'inner_val_groups':len(group_sets[1]),
            'outer_test_groups':len(group_sets[2]),'outer_test_images':len(test),
            **{f'test_{k}':v for k,v in metrics_to_dict(test_metrics).items()},
            'identity':identity}
    json_write(save_dir/'fold_metrics.json',result)  # Written LAST: completion sentinel
    print(f'COMPLETE {a.model} fold={a.fold} outer_test_group_acc={test_metrics.group_accuracy:.4f}')


def metrics_from_cm(cm):
    cm=np.asarray(cm,dtype=int)
    support=cm.sum(1)
    preds=cm.sum(0)
    correct=np.diag(cm)
    total=int(cm.sum())
    recalls=np.divide(correct,support,out=np.zeros(3,dtype=float),where=support!=0)
    precisions=np.divide(correct,preds,out=np.zeros(3,dtype=float),where=preds!=0)
    f1=np.divide(2*precisions*recalls,precisions+recalls,
                 out=np.zeros(3,dtype=float),where=(precisions+recalls)!=0)
    return {'accuracy':float(correct.sum()/total),
            'balanced_accuracy':float(recalls.mean()),
            'macro_f1':float(f1.mean()),
            'n':total,'confusion_matrix':cm.tolist(),
            'by_class':{name:{'support':int(support[i]),'recall':float(recalls[i]),
                              'precision':float(precisions[i]),'f1':float(f1[i])}
                        for i,name in enumerate(LABELS)}}


def summarize(a):
    # No model selection/ranking is done using test metrics.
    metadata = json.loads((Path(a.out)/'run_metadata.json').read_text())
    role_path = Path(a.out)/'role_assignments.csv'
    if sha256(role_path) != metadata['roles_sha256']:
        raise RuntimeError('Frozen split plan was modified after preparation')
    roles = pd.read_csv(role_path)
    all_groups = set(roles.group_key)
    models=[]; fold_rows=[]
    for name in MODELS:
        group_dfs=[]; image_dfs=[]
        for fold in range(1,FOLD_COUNT+1):
            d=Path(a.out)/'models'/name/f'fold_{fold}'
            mpath=d/'fold_metrics.json';gpath=d/'test_group_predictions.csv';ipath=d/'test_image_predictions.csv.gz'
            if not all(x.is_file() for x in (mpath,gpath,ipath)):
                raise RuntimeError(f'Missing complete heldout fold: {name} {fold}')
            m=json.loads(mpath.read_text())
            expected={x for x in roles.loc[(roles.fold==fold)&(roles.role=='test'),'group_key']}
            g=pd.read_csv(gpath); img=pd.read_csv(ipath)
            if set(g.group_key)!=expected or g.group_key.duplicated().any():
                raise RuntimeError(f'Wrong outer heldout groups: {name} {fold}')
            if set(img.group_key)!=expected or len(img)!=int(roles.loc[(roles.fold==fold)&(roles.role=='test'),'n_images'].sum()):
                raise RuntimeError(f'Wrong heldout images: {name} {fold}')
            identity = m['identity']
            if (m['outer_test_used_for_checkpoint_selection'] is not False or
                m['model'] != name or m['fold'] != fold or
                identity['roles_sha256'] != metadata['roles_sha256'] or
                identity['manifest_sha256'] != metadata['manifest_sha256'] or
                identity['trainer_sha256'] != sha256(__file__) or
                identity['shared_code_sha256'] != sha256(Path(__file__).with_name('dl_common.py'))):
                raise RuntimeError(f'Fold selection/leakage/code identity contract failed: {name} {fold}')
            expected_class = roles.set_index('group_key').target.to_dict()
            if any(int(row.true_index) != int(expected_class[row.group_key]) for row in g.itertuples()):
                raise RuntimeError(f'Fold labels do not match frozen split: {name} {fold}')
            group_dfs.append(g); image_dfs.append(img)
            fold_rows.append({'model':name,'fold':fold,'best_epoch':m['best_epoch'],
                              'train_groups':m['train_groups'],'inner_val_groups':m['inner_val_groups'],
                              'test_groups':m['outer_test_groups'],
                              'test_group_accuracy':m['test_group_accuracy'],
                              'test_group_balanced_accuracy':m['test_group_balanced_accuracy'],
                              'test_group_macro_f1':m['test_group_macro_f1'],
                              'test_image_accuracy':m['test_image_accuracy'],
                              'test_image_balanced_accuracy':m['test_image_balanced_accuracy'],
                              'test_image_macro_f1':m['test_image_macro_f1']})
        oof=pd.concat(group_dfs,ignore_index=True); iof=pd.concat(image_dfs,ignore_index=True)
        if len(oof)!=len(all_groups) or oof.group_key.nunique()!=len(all_groups) or set(oof.group_key)!=all_groups:
            raise RuntimeError(f'Not exactly one outer test per subgroup: {name}')
        if len(iof)!=int(roles.loc[roles.fold==1,'n_images'].sum()) or iof.image_path.duplicated().any():
            raise RuntimeError(f'Not exactly one outer test per angle image: {name}')
        group_cm=confusion_matrix(oof.true_index,oof.pred_index,labels=range(3))
        image_cm=confusion_matrix(iof.true_index,iof.pred_index,labels=range(3))
        model_rows=[x for x in fold_rows if x['model']==name]
        def stats(metric):
            vals=np.array([float(x[metric]) for x in model_rows])
            return {'mean':float(vals.mean()),'std':float(vals.std(ddof=1))}
        models.append({'model':name,'n_folds':5,'pooled_group':metrics_from_cm(group_cm),
                       'pooled_image':metrics_from_cm(image_cm),
                       'fold_mean_group_accuracy':stats('test_group_accuracy'),
                       'fold_mean_group_macro_f1':stats('test_group_macro_f1'),
                       'fold_mean_image_accuracy':stats('test_image_accuracy'),
                       'note':'Final tests are never consulted for checkpoint selection'})
    pd.DataFrame(fold_rows).to_csv(Path(a.out)/'fold_results.csv',index=False)
    json_write(Path(a.out)/'heldout_5fold_summary.json',
               {'status':'complete','n_independent_subgroups':len(all_groups),
                'label_order':LABELS,'models':models,
                'design':'5 outer test folds + inner validation by subgroup; no test-driven model selection',
                'grouping_limit':'No biological identity verification for distinct group keys'})
    print(json.dumps({'status':'complete','models':[{'model':m['model'],
                 'pooled_group':m['pooled_group']} for m in models]},indent=2))


def get_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','summarize'))
    p.add_argument('--manifest',default='manifest_clean.csv')
    p.add_argument('--out',default='./output/angle_cnn')
    p.add_argument('--model',choices=MODELS)
    p.add_argument('--fold',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--inner-val-fraction',type=float,default=0.2)
    p.add_argument('--image-size',type=int,default=224)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--num-workers',type=int,default=8)
    p.add_argument('--max-epochs',type=int,default=50)
    p.add_argument('--head-epochs',type=int,default=8)
    p.add_argument('--early-stop-patience',type=int,default=8)
    p.add_argument('--min-epochs',type=int,default=12)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--finetune-head-lr',type=float,default=1e-4)
    p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--dropout',type=float,default=.3)
    p.add_argument('--label-smoothing',type=float,default=.05)
    p.add_argument('--allow-cpu',action='store_true',help='Development only; not for final GPU campaign')
    a=p.parse_args(argv)
    if a.action=='run' and (a.model is None or a.fold is None):
        p.error('run requires --model and --fold')
    if a.max_epochs<a.min_epochs or a.min_epochs<1 or a.early_stop_patience<1:
        p.error('Expected max_epochs >= min_epochs >= 1, patience >= 1')
    return a


if __name__=='__main__':
    args=get_args()
    if args.action=='prepare':prepare(args)
    elif args.action=='run':train_one(args)
    else:summarize(args)