#!/usr/bin/env python3
"""Create isolated, auditable full-frame CNN repeat sources; never edits historical code/results.

Generate a copy of the original validated trainers, changing *only* the image
transforms and the model registry. Keep all original split, validation,
checkpoint, prediction, and result contract logic.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

MODELS = ('resnet18','efficientnet_b0','mobilenet_v3_large')

CORE_TRANSFORMS = '''def make_transforms(train: bool, image_size: int = 224):
    # Preserve the full original field of view and its aspect ratio, including borders.
    # PIL ImageOps.pad scales until it fits; the remainder is neutral gray padding.
    from PIL import ImageOps
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    preprocess = transforms.Lambda(
        lambda im: ImageOps.pad(im, (image_size, image_size),
                                method=Image.Resampling.BICUBIC,
                                color=(127, 127, 127), centering=(0.5, 0.5)))
    if train:
        return transforms.Compose([preprocess,
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.10, 0.10, 0.08, 0.02),
            transforms.ToTensor(), norm])
    return transforms.Compose([preprocess, transforms.ToTensor(), norm])
'''

CORE_MODELS = '''def build_model(name,n_classes,pretrained=True):
    n=name.lower()
    if n=="resnet18":
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); m.fc=nn.Linear(m.fc.in_features,n_classes); m._head_module=m.fc; return m
    if n in {"efficientnet_b0","efficientnet-b0"}:
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,n_classes); m._head_module=m.classifier; return m
    if n in {"mobilenet_v3_large","mobilenetv3_large","mobilenetv3-large"}:
        m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None); m.classifier[3]=nn.Linear(m.classifier[3].in_features,n_classes); m._head_module=m.classifier; return m
    raise ValueError(name)'''

ANGLE_TRAIN = '''def train_transform(sz=224):
    # Keep the same full-frame FOV as inference. Training randomness is only
    # horizontal/vertical? No flips here: microscopy angle geometry might matter.
    from PIL import ImageOps
    full = transforms.Lambda(lambda im: ImageOps.pad(
        im, (sz, sz), method=Image.Resampling.BICUBIC,
        color=(127, 127, 127), centering=(0.5, 0.5)))
    return transforms.Compose([full,
        transforms.ColorJitter(brightness=.1, contrast=.1),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
'''

ANGLE_EVAL = '''def eval_transform(sz=224):
    from PIL import ImageOps
    full = transforms.Lambda(lambda im: ImageOps.pad(
        im, (sz, sz), method=Image.Resampling.BICUBIC,
        color=(127, 127, 127), centering=(0.5, 0.5)))
    return transforms.Compose([full, transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
'''

ANGLE_MODELS = '''def build_model(name,num_classes=3,dropout=.3,pretrained=True):
    n=name.lower()
    if n=='resnet18':
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); f=m.fc.in_features; m.fc=nn.Sequential(nn.Dropout(dropout),nn.Linear(f,num_classes)); return m
    if n in {'efficientnet_b0','efficientnet-b0'}:
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None); f=m.classifier[-1].in_features; m.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(f,num_classes)); return m
    if n in {'mobilenet_v3_large','mobilenetv3_large'}:
        m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None); f=m.classifier[-1].in_features; m.classifier[-1]=nn.Linear(f,num_classes); return m
    raise ValueError(name)'''

ANGLE_SPLIT = '''def split_parameter_groups(model, name):
    head = list(model.fc.parameters()) if name.lower() == 'resnet18' else list(model.classifier.parameters())
    ids = {id(p) for p in head}
    back = [p for p in model.parameters() if id(p) not in ids]
    return back, head
'''

def replace_function(text: str, name: str, replacement: str) -> str:
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    matches = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name == name]
    if len(matches)!=1:
        raise RuntimeError(f'Expected exactly 1 {name}; found {len(matches)}')
    node=matches[0]
    assert node.col_offset == 0
    lines[node.lineno-1:node.end_lineno] = [replacement.rstrip()+'\n']
    return ''.join(lines)

def replace_exact(text: str, old: str, new: str, path: str) -> str:
    if text.count(old)!=1: raise RuntimeError(f'{path}: expected exactly one occurrence of {old!r}; got {text.count(old)}')
    return text.replace(old,new)

def copy_and_edit(src: Path, dst: Path, fn=None):
    if not src.is_file(): raise FileNotFoundError(src)
    txt=src.read_text()
    if fn: txt=fn(txt)
    ast.parse(txt, filename=str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(txt)
    return {'file':str(dst.relative_to(dst.parents[2])), 'sha256':hashlib.sha256(txt.encode()).hexdigest()}

def build(repo: Path, dest: Path, overwrite: bool=False):
    generated=dest/'generated'
    if generated.exists():
        if not overwrite: raise FileExistsError(f'{generated} exists; rerun with --overwrite-sources only after checking no jobs are using it')
        shutil.rmtree(generated)
    core=generated/'core'; angle=generated/'angle'
    records=[]
    for name in ['cnn_checkpoint_state.py','core_eval.py','run_contract.py']:
        records.append(copy_and_edit(repo/'experiment1_tissue/core_cnn_b0_a1'/name,core/name))
    def edit_core_common(s):
        s=replace_function(s,'make_transforms',CORE_TRANSFORMS)
        return replace_function(s,'build_model',CORE_MODELS)
    records.append(copy_and_edit(repo/'experiment1_tissue/core_cnn_b0_a1'/'dl_grouped_common.py',core/'dl_grouped_common.py',edit_core_common))
    old="choices=['resnet18','efficientnet_b0','mobilenet_v3_large']"
    models_str='choices='+repr(list(MODELS))
    records.append(copy_and_edit(repo/'experiment1_tissue/core_cnn_b0_a1'/'run_one_core_cnn_fold_resumable.py',core/'run_one_core_cnn_fold_resumable.py',
                                 lambda s:replace_exact(s,old,models_str,'core runner')))
    def edit_angle_common(s):
        for name,body in [('train_transform',ANGLE_TRAIN),('eval_transform',ANGLE_EVAL),
                          ('build_model',ANGLE_MODELS),('split_parameter_groups',ANGLE_SPLIT)]:
            s=replace_function(s,name,body)
        return s
    records.append(copy_and_edit(repo/'experiment2_angle_sweep_cells/angle_cnn/dl_common.py',angle/'dl_common.py',edit_angle_common))
    original="MODELS = ('resnet18', 'efficientnet_b0', 'mobilenet_v3_large')"
    records.append(copy_and_edit(repo/'experiment2_angle_sweep_cells/angle_cnn/angle_cnn_heldout_cv.py',angle/'angle_cnn_heldout_cv.py',
                                 lambda s:replace_exact(s,original,'MODELS = '+repr(MODELS),'angle runner')))
    jobrows=[]
    for experiment,folds in [('B0',(1,2,3,4,5)),('A1',(2,4))]:
        for sc in ('scenario1','scenario2'):
            for model in MODELS:
                for fold in folds:
                    jobrows.append(('core',experiment,sc,model,fold))
    for model in MODELS:
        for fold in range(1,6): jobrows.append(('angle','ANGLE','scenario3',model,fold))
    (dest/'jobs.tsv').write_text(''.join('\t'.join(map(str,r))+'\n' for r in jobrows))
    (dest/'build_manifest.json').write_text(json.dumps({
        'experiment':'FULLFRAME_V1','models':MODELS,'n_jobs':len(jobrows),
        'core_B0_jobs':30,'core_A1_jobs':12,'angle_jobs':15,
        'baseline_kept':'All original scripts and results are left unchanged',
        'training_policy':'Original trained ImageNet backbones, head-only warmup then full fine-tuning',
        'preprocess':'224 RGB, preserve original full-frame and aspect ratio using ImageOps.pad (gray letterbox), consistent across train/validation/test',
        'code_hashes':records
    },indent=2))
    return len(jobrows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--overwrite-sources',action='store_true')
    a=p.parse_args()
    print(f'Generated {build(a.repo,a.output,a.overwrite_sources)} training jobs at {a.output}')
