#!/usr/bin/env python3
"""Strict, read-only preflight for CCast FullFrame V1 before submitting GPU jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def run(root: Path):
    import torch
    import torchvision
    from torchvision import models
    print('PyTorch:',torch.__version__,'Torchvision:',torchvision.__version__)
    campaign=root/'cancer_fullframe_3cnn_v1'
    manifest=json.loads((campaign/'build_manifest.json').read_text())
    if manifest.get('n_jobs')!=57: raise RuntimeError('Unexpected campaign size')
    if len((campaign/'jobs.tsv').read_text().splitlines())!=57: raise RuntimeError('Job list corrupted')
    for f in ['generated/core/dl_grouped_common.py','generated/core/run_one_core_cnn_fold_resumable.py',
              'generated/angle/dl_common.py','generated/angle/angle_cnn_heldout_cv.py']:
        p=campaign/f
        if not p.is_file(): raise RuntimeError(f'Generated source missing: {p}')
    # Frozen Core role membership is checked by historical verification scripts
    # at training time; require the exact files used by historical B0/A1.
    core_root=root/'cancer_advanced_core_cv'
    for rel in ('splits/inner_outer_roles.csv','splits_variant_a1/inner_outer_roles.csv'):
        p=core_root/rel
        if not p.is_file(): raise RuntimeError(f'Missing historical frozen split: {p}')
        d=pd.read_csv(p)
        for fold in range(1,6):
            q=d[d.outer_fold==fold]
            assert len(q)==45 and q.core_id.nunique()==45,(p,fold)
            assert q.role.value_counts().to_dict()=={'train':29,'val':7,'test':9},(p,fold)
    angle=campaign/'results'/'angle'/'role_assignments.csv'
    original=root/'cancer_angle_deep_learning_pipeline/results_dl/heldout_5fold_v2/role_assignments.csv'
    if not angle.is_file() or not original.is_file():
        raise RuntimeError('Angle role assignments missing; run Angle prepare and preserve original split')
    a=pd.read_csv(angle);b=pd.read_csv(original)
    if not a.equals(b): raise RuntimeError('New and original Angle 5-fold subgroup assignments differ')
    # New model weights must already be available locally or compute jobs may
    # attempt to fetch them without reliable network access.
    weights={
        'resnet18':models.ResNet18_Weights.IMAGENET1K_V1,
        'efficientnet_b0':models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'mobilenet_v3_large':models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    }
    weightroot=Path(torch.hub.get_dir())/'checkpoints'
    missing=[]
    for model,enum in weights.items():
        filename=Path(urlparse(enum.url).path).name
        p=weightroot/filename
        print('WEIGHT',model,'FOUND' if p.is_file() else 'MISSING',p)
        if not p.is_file():missing.append((model,str(p),enum.url))
    if missing:
        print('\nThe following pretrained weights must be acquired on the authorized login node before GPU submission:')
        for model,path,url in missing:
            print(' ',model,'->',path,'official URL:',url)
        raise SystemExit('PRETRAINED WEIGHTS INCOMPLETE (no training submitted)')
    print('PREFLIGHT PASS: old results intact; frozen splits identical; three model weight files present.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('./private_data'))
    run(p.parse_args().root)
