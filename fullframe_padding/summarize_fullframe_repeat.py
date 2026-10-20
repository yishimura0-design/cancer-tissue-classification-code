#!/usr/bin/env python3
"""Pool complete Core predictions without ever mixing B0/A1 or image/core units."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

MODELS=('resnet18','efficientnet_b0','mobilenet_v3_large')

def metrics(df, classes, target_col, pred_col):
    y=df[target_col].to_numpy(dtype=int)
    p=df[pred_col].to_numpy(dtype=int)
    assert np.isin(y,range(len(classes))).all()
    assert np.isin(p,range(len(classes))).all()
    cm=confusion_matrix(y,p,labels=range(len(classes)))
    return {'overall_accuracy':float(accuracy_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=list(range(len(classes))),average='macro',zero_division=0)),
            'n':len(y),'confusion_matrix':cm.tolist(),
            'recall_by_class':{cl:float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None for i,cl in enumerate(classes)},
            'support_by_class':{cl:int(cm[i].sum()) for i,cl in enumerate(classes)}}

def run(root):
    rows=[]
    for exp in ('B0','A1'):
        for sc in ('scenario1','scenario2'):
            for model in MODELS:
                parts=[];iparts=[];metadata=[]
                for fold in range(1,6):
                    source='B0' if exp=='A1' and fold in (1,3,5) else exp
                    d=root/'results'/'core'/source/sc/model/f'fold_{fold}'
                    expected=(d/'fold_metrics.json',d/'test_core_predictions.csv',d/'test_image_predictions.csv.gz',d/'run_contract.json')
                    if any(not p.is_file() for p in expected):
                        raise FileNotFoundError(f'INCOMPLETE: {exp} {sc} {model} fold {fold}: {d}')
                    m=json.loads((d/'fold_metrics.json').read_text())
                    ident=json.loads((d/'run_contract.json').read_text())
                    if ident!=m.get('run_contract') or ident['config']['experiment_id']!=source or m['fold']!=fold:
                        raise RuntimeError(f'Fingerprint mismatch: {d}')
                    classes=m['classes']
                    core=pd.read_csv(d/'test_core_predictions.csv')
                    image=pd.read_csv(d/'test_image_predictions.csv.gz')
                    if len(core)!=9 or len(image)!=3240: raise RuntimeError(f'Incomplete heldout coverage: {d}')
                    if core.core_id.nunique()!=9 or image.core_id.nunique()!=9:
                        raise RuntimeError(f'Non-unique core membership: {d}')
                    parts.append(core);iparts.append(image);metadata.append(str(d))
                C=pd.concat(parts,ignore_index=True); I=pd.concat(iparts,ignore_index=True)
                if C.core_id.nunique()!=45 or len(C)!=45 or len(I)!=16200:
                    raise RuntimeError(f'OOF coverage invalid: {exp} {sc} {model}')
                if C.core_id.duplicated().any() or I.path.duplicated().any():
                    raise RuntimeError('Repeated Test identifiers')
                if I.groupby('core_id').size().ne(360).any():
                    raise RuntimeError('Uneven per-Core image count')
                # Core output provides target and prediction; image provides target and prediction?
                # 'target' and 'prediction' are checked explicitly, no guessed labels.
                for unit,d in [('core',C),('image',I)]:
                    tgt='target' if 'target' in d else 'y_true'
                    pred='prediction' if 'prediction' in d else 'pred_index'
                    if tgt not in d or pred not in d:
                        raise RuntimeError(f'Missing numeric {unit} prediction columns: {list(d.columns)}')
                    rows.append({'experiment':exp,'scenario':sc,'model':model,'unit':unit,
                                 **metrics(d,classes,tgt,pred),'sources':metadata})
    out=root/'results'/'pooled_core_fullframe.json'
    out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(rows,indent=2))
    print(f'COMPLETE: {len(rows)} pooled model/experiment/scenario/unit summaries -> {out}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    run(p.parse_args().campaign)
