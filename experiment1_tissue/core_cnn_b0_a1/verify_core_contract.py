#!/usr/bin/env python3
"""Fail-closed audit of all accepted TIFFs against shared 5-fold core roles.

No image loading: runs on manifest metadata. Validation must be nonempty;
individual classes may be absent from validation or test. Both scenarios share exactly the same outer-test core IDs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

CORE='sample_id_from_folder'


def verify(manifest_csv: str, roles_csv: str) -> dict:
    raw=pd.read_csv(manifest_csv)
    raw=raw[raw.qc_status.eq('accepted_tif')].copy() if 'qc_status' in raw else raw.copy()
    roles=pd.read_csv(roles_csv)
    required_cols={'relative_path', 'subgroup_id', CORE,'scenario1_label','scenario2_label'}
    if not required_cols.issubset(raw.columns):
        raise ValueError(f'Missing columns: {required_cols-set(raw.columns)}')
    if raw[CORE].isna().any():
        raise ValueError('Null core ID')
    raw[CORE]=raw[CORE].astype(str).str.upper().str.strip()
    groups=raw.groupby(CORE)
    if len(raw)!=16200 or raw[CORE].nunique()!=45:
        raise ValueError('Expected 16,200 images and 45 distinct cores')
    if (groups.size()!=360).any() or (groups.subgroup_id.nunique()!=5).any():
        raise ValueError('Each core must contain five 72-angle fields = 360 images')
    if (raw.groupby('subgroup_id').size()!=72).any() or raw.subgroup_id.nunique()!=225:
        raise ValueError('Every field must have exactly 72 angular images')
    for col in ['scenario1_label','scenario2_label']:
        if (groups[col].nunique()!=1).any():
            raise ValueError(f'Inconsistent core label in {col}')
    base_cores=set(raw[CORE]); test_membership=[]; report=[]
    if set(pd.to_numeric(roles.outer_fold).unique())!=set(range(1,6)):
        raise ValueError('Exactly 5 outer folds are required')
    for fold, part in roles.groupby('outer_fold'):
        if len(part)!=45 or part.core_id.duplicated().any() or set(part.core_id)!=base_cores:
            raise ValueError(f'Incomplete or repeated core membership fold={fold}')
        if not set(part.role).issubset({'train','val','test'}):
            raise ValueError(f'Unknown split role fold={fold}')
        train=set(part.loc[part.role=='train','core_id'])
        val=set(part.loc[part.role=='val','core_id'])
        test=set(part.loc[part.role=='test','core_id'])
        if not train or not test or not val:
            raise ValueError(f'Fold {fold}: train, validation and test must be nonempty')
        if train & val or train & test or val & test:
            raise ValueError(f'Fold {fold}: BIOLOGICAL CORE LEAKAGE')
        if train|val|test!=base_cores:
            raise ValueError(f'Fold {fold}: unassigned cores')
        # Validate *image* allocation, not just core IDs.
        joined=raw[[CORE,'subgroup_id','relative_path']].merge(part[['core_id','role']],left_on=CORE,right_on='core_id',validate='many_to_one')
        if len(joined)!=16200 or joined['role'].isna().any():
            raise ValueError('Image-to-role assignment incomplete')
        if (joined.groupby('subgroup_id').role.nunique()!=1).any():
            raise ValueError('One angular field split across roles')
        if (joined.groupby(CORE).role.nunique()!=1).any():
            raise ValueError('One tissue core split across roles')
        for task in ['scenario1_label','scenario2_label']:
            if set(part.loc[part.role=='train',task])!=set(raw[task]):
                raise ValueError(f'Fold {fold}: train missing labels in {task}')
        test_membership.extend(test)
        report.append({'fold':int(fold),'train_cores':len(train),'val_cores':len(val),'test_cores':len(test),
                       'train_images':len(joined[joined.role=='train']),
                       'val_images':len(joined[joined.role=='val']),
                       'test_images':len(joined[joined.role=='test']),
                       'test_stage_counts':part[part.role=='test'].scenario2_label.value_counts().to_dict(),
                       'val_stage_counts':part[part.role=='val'].scenario2_label.value_counts().to_dict()})
    if len(test_membership)!=45 or len(set(test_membership))!=45:
        raise ValueError('Outer test must cover every core EXACTLY ONCE')
    a1=raw[raw[CORE]=='A1'].subgroup_id.unique()
    if len(a1)!=5 or not any('A1_1_' in str(x) for x in a1) or not any('A1_2_' in str(x) for x in a1):
        raise ValueError('A1 field identity audit failed')
    return {'status':'PASS','folds':report,'n_cores':45,'n_fields':225,'n_images':16200,
            'binary_task':'Normal vs Malignant','stage_task':'6-class tissue stages',
            'outer_test_once_per_core':True,'allowed_zero_validation':False,
            'patient_independence_verified':False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--roles',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); report=verify(a.manifest,a.roles)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PASS: all 45 biological cores isolated in 5 held-out test folds; individual rare stages may be absent from validation')

if __name__=='__main__':main()
