#!/usr/bin/env python3
"""Create leakage-safe tissue-core 5-fold outer and inner splits.

Input: original QC manifest (16,200 image records with sample_id_from_folder).
All five fields of view and their 72 angular images share a single core fold.
Sparse classes (2 cores) are *not* forced into inner validation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

STAGE_COL = 'scenario2_label'
BINARY_COL = 'scenario1_label'
CORE_COL = 'sample_id_from_folder'


def core_table_from_manifest(path: str, check_profile: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'qc_status' in df:
        df = df[df['qc_status'].astype(str).str.startswith('accepted')].copy()
    need = {CORE_COL, STAGE_COL, BINARY_COL, 'subgroup_id', 'relative_path'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'Manifest missing columns: {sorted(missing)}')
    if df[list(need)].isna().any().any():
        raise ValueError('Null identifiers or labels in the accepted manifest')
    df[CORE_COL] = df[CORE_COL].astype(str).str.upper().str.strip()
    for c in [STAGE_COL, BINARY_COL]:
        inconsistent = df.groupby(CORE_COL)[c].nunique()
        if (inconsistent != 1).any():
            raise ValueError(f'Core has more than one {c}: {inconsistent[inconsistent!=1].to_dict()}')
    fields = df.groupby(CORE_COL)['subgroup_id'].nunique()
    sizes = df.groupby(CORE_COL).size()
    if check_profile:
        if not (len(df) == 16200 and len(sizes) == 45 and (sizes == 360).all() and (fields == 5).all()):
            raise ValueError(f'Unexpected clean dataset: images={len(df)}, cores={len(sizes)}, core_sizes={sorted(sizes.unique())}, fields={sorted(fields.unique())}')
        angle_counts = df.groupby('subgroup_id').size()
        if len(angle_counts) != 225 or not (angle_counts == 72).all():
            raise ValueError('Expected 225 fields of view x 72 angles')
    cores = df.groupby(CORE_COL, sort=True).agg(
        scenario1_label=(BINARY_COL, 'first'),
        scenario2_label=(STAGE_COL, 'first'),
        n_images=(CORE_COL, 'size'),
        n_fields=('subgroup_id', 'nunique'),
    ).reset_index().rename(columns={CORE_COL:'core_id'})
    return cores


def assign_outer_folds(cores: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError('At least two outer folds are required')
    rng = np.random.default_rng(seed)
    out = cores.copy()
    out['fold'] = 0
    occupancy = np.zeros(n_folds, dtype=int)
    stage_counts = {v: np.zeros(n_folds, dtype=int) for v in out['scenario2_label'].unique()}
    class_counts = Counter(out['scenario2_label'])
    # Rare classes first, so the two available cores land in distinct folds.
    for label in sorted(class_counts, key=lambda x:(class_counts[x], x)):
        indices = out.index[out['scenario2_label']==label].to_numpy()
        rng.shuffle(indices)
        for ix in indices:
            permitted = np.flatnonzero(stage_counts[label] == stage_counts[label].min())
            min_total = occupancy[permitted].min()
            permitted = permitted[occupancy[permitted] == min_total]
            chosen = int(rng.choice(permitted))
            out.loc[ix,'fold'] = chosen + 1
            occupancy[chosen] += 1
            stage_counts[label][chosen] += 1
    if (out['fold'] == 0).any():
        raise RuntimeError('Unassigned outer core')
    for label, count in class_counts.items():
        if count >= 2 and max(stage_counts[label]) - min(stage_counts[label]) > 1:
            raise RuntimeError(f'Unbalanced class distribution: {label}')
        if count == 2 and max(stage_counts[label]) > 1:
            raise RuntimeError(f'Both rare-class cores assigned to same test fold: {label}')
    for f in range(1,n_folds+1):
        train_labels = set(out.loc[out.fold != f,'scenario2_label'])
        if train_labels != set(class_counts):
            raise RuntimeError(f'Fold {f} training missing classes: {set(class_counts)-train_labels}')
    return out.sort_values('core_id').reset_index(drop=True)


def inner_validation_cores(outer_train: pd.DataFrame, frac: float, seed: int) -> set[str]:
    """Hold out common-stage cores, preserving ALL Stage III/IV cores for training.

    Labels with <=2 outer-training cores are not eligible for validation.
    Select ~frac of outer-training cores (usually 7), distributing common stages.
    """
    if frac <= 0:
        raise ValueError('Nonempty inner validation is required')
    rng = np.random.default_rng(seed)
    stage_groups = {}
    for label, g in outer_train.groupby('scenario2_label'):
        ids = g['core_id'].tolist()
        rng.shuffle(ids)
        if len(ids) >= 3:
            stage_groups[label] = ids
    if not stage_groups:
        raise ValueError('Cannot allocate a nonempty inner validation split')
    target = max(len(stage_groups), round(frac*len(outer_train)))
    quota = {k:1 for k in stage_groups}
    target = min(target, sum(len(v)-1 for v in stage_groups.values()))
    while sum(quota.values()) < target:
        eligible = [k for k in stage_groups if quota[k] < len(stage_groups[k])-1]
        if not eligible:
            break
        # Larger groups receive proportionally more validation cores.
        label = max(eligible, key=lambda k:(len(stage_groups[k])*frac-quota[k],len(stage_groups[k]),k))
        quota[label]+=1
    return {core for label, q in quota.items() for core in stage_groups[label][:q]}


def build_role_table(outer: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42) -> pd.DataFrame:
    rows = []
    for f in sorted(outer.fold.unique()):
        tr = outer[outer.fold != f]
        val = inner_validation_cores(tr, val_fraction, seed+1000*int(f))
        for row in outer.itertuples(index=False):
            role = 'test' if row.fold == f else ('val' if row.core_id in val else 'train')
            rows.append({'outer_fold':int(f),'core_id':row.core_id,'role':role,
                         'scenario1_label':row.scenario1_label,'scenario2_label':row.scenario2_label,
                         'n_images':int(row.n_images),'n_fields':int(row.n_fields)})
    roles = pd.DataFrame(rows)
    for f, part in roles.groupby('outer_fold'):
        buckets = {r:set(part.loc[part.role==r,'core_id']) for r in ['train','val','test']}
        assert all(buckets[a].isdisjoint(buckets[b]) for a,b in [('train','val'),('train','test'),('val','test')])
        assert sum(map(len,buckets.values())) == len(outer)
        if not buckets['train'] or not buckets['test'] or not buckets['val']:
            raise ValueError(f'Empty train/validation/test split in outer fold {f}')
        tr_classes = set(part.loc[part.role=='train','scenario2_label'])
        if tr_classes != set(outer.scenario2_label):
            raise ValueError(f'Inner train missing classes in outer fold {f}: {set(outer.scenario2_label)-tr_classes}')
        if len(part.loc[(part.role=='val') & part.scenario2_label.isin(['Stage III','Stage IV'])]):
            raise ValueError('Rare-class validation unexpectedly nonzero')
    return roles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--inner-val-fraction',type=float,default=.2)
    a=ap.parse_args()
    if not (0<a.inner_val_fraction<.5):
        raise ValueError('Use inner-val-fraction greater than zero and less than 0.5')
    out_dir=Path(a.out)
    out_dir.mkdir(parents=True,exist_ok=True)
    core=core_table_from_manifest(a.manifest)
    outer=assign_outer_folds(core,a.folds,a.seed)
    roles=build_role_table(outer,a.inner_val_fraction,a.seed)
    outer.to_csv(out_dir/'outer_core_folds.csv',index=False)
    roles.to_csv(out_dir/'inner_outer_roles.csv',index=False)
    summary={
        'n_cores':len(core),'n_images':int(core.n_images.sum()),
        'class_counts':core.scenario2_label.value_counts().to_dict(),
        'fold_sizes':outer.fold.value_counts().sort_index().to_dict(),
        'per_fold_stage_counts':pd.crosstab(outer.fold,outer.scenario2_label).to_dict(orient='index'),
        'per_fold_roles':pd.crosstab(roles.outer_fold,roles.role).to_dict(orient='index'),
        'inner_val_fraction': a.inner_val_fraction,
        'validation_empty': False,
        'scenario1_class_counts':core.scenario1_label.value_counts().to_dict(),
        'warning':'Only two Stage III and two Stage IV cores; outer tests need not contain all classes. Nonempty inner validation is mandatory for checkpoint/model selection. Splits test unseen CORE IDs, not patient-level independence.',
    }
    (out_dir/'split_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
