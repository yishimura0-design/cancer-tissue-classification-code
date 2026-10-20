#!/usr/bin/env python3
"""Predeclared sensitivity experiment: swap C2 and C3 outer TEST ownership.

Baseline: C2 is test in fold 2 and C3 is test in fold 4.
Variant A1: C3 is test in fold 2 and C2 is test in fold 4.
All seven validation cores per fold remain identical. This is NOT a new cohort.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd
from verify_core_contract import verify

SWAPS={2:('C2','C3'),4:('C3','C2')}

def make_variant(base:pd.DataFrame)->pd.DataFrame:
    result=base.copy()
    must={'outer_fold','core_id','role','scenario2_label'}
    if not must.issubset(base.columns):
        raise ValueError(f'Missing roles columns: {must-set(base.columns)}')
    for fold,(was_test,was_train) in SWAPS.items():
        original=base[base.outer_fold==fold].set_index('core_id')
        if original.at[was_test,'role'] != 'test' or original.at[was_train,'role'] != 'train':
            raise ValueError(f'Predeclared swap not applicable: fold {fold}; {was_test}=test, {was_train}=train required')
        mask1=(result.outer_fold==fold)&(result.core_id==was_test)
        mask2=(result.outer_fold==fold)&(result.core_id==was_train)
        result.loc[mask1,'role']='train'
        result.loc[mask2,'role']='test'
    for fold in range(1,6):
        b=base[base.outer_fold==fold].set_index('core_id')
        v=result[result.outer_fold==fold].set_index('core_id')
        for k in ['train','val','test']:
            if (b.role==k).sum() != (v.role==k).sum():
                raise AssertionError('Split sizes must be preserved')
        if set(b.index[b.role=='val']) != set(v.index[v.role=='val']):
            raise AssertionError('Validation set changed')
        if fold not in SWAPS and not b.equals(v):
            raise AssertionError('Unchanged fold differs')
        if len(v.index[v.role=='train'])!=29 or len(v.index[v.role=='val'])!=7 or len(v.index[v.role=='test'])!=9:
            raise AssertionError('Expected 29/7/9 core split')
        if set(v.loc[v.role=='train','scenario2_label'])!=set(base.scenario2_label):
            raise AssertionError('Training class missing')
    return result

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-roles',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    source=pd.read_csv(a.baseline_roles)
    variant=make_variant(source)
    target=dest/'inner_outer_roles.csv'
    variant.to_csv(target,index=False)
    evidence=verify(a.manifest,str(target))
    assert evidence['status']=='PASS'
    report={'experiment_id':'A1','design':'exploratory_core_fold_pattern',
            'baseline_roles_sha256':sha(a.baseline_roles),'variant_roles_sha256':sha(target),
            'changed_folds':[2,4],'unchanged_folds':[1,3,5],
            'swaps':{str(f):{'test_to_train':a,'train_to_test':b} for f,(a,b) in SWAPS.items()},
            'leakage_audit':evidence}
    (dest/'variant_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='leakage_audit'},indent=2))
if __name__=='__main__': main()
