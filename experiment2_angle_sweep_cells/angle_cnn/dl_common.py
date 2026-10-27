from __future__ import annotations
import json, random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import models, transforms
CLASS_ORDER=['HPNE','PANC','MIA']
CLASS_TO_INDEX={c:i for i,c in enumerate(CLASS_ORDER)}

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

def detect_columns(df):
    p=next((c for c in ['path','filepath','file_path','image_path'] if c in df.columns),None)
    l=next((c for c in ['label','class','target'] if c in df.columns),None)
    g=next((c for c in ['subgroup_key','group_key','group','sample_id'] if c in df.columns),None)
    if None in (p,l,g): raise ValueError(f'Required columns not found: {list(df.columns)}')
    return p,l,g

def load_manifest(path):
    df=pd.read_csv(path); p,l,g=detect_columns(df); df=df.copy()
    df['image_path']=df[p].astype(str); df['label']=df[l].astype(str).str.upper(); df['group_key']=df[g].astype(str)
    df=df[df['label'].isin(CLASS_ORDER)].reset_index(drop=True); df['target']=df['label'].map(CLASS_TO_INDEX).astype(int)
    missing=[x for x in df.image_path if not Path(x).is_file()]
    if missing: raise FileNotFoundError(f'{len(missing)} missing images, first: {missing[0]}')
    return df

def train_transform(sz=224):
    return transforms.Compose([transforms.RandomResizedCrop(sz,scale=(.85,1.0),ratio=(.95,1.05)),transforms.RandomAffine(degrees=0,translate=(.05,.05)),transforms.ColorJitter(brightness=.1,contrast=.1),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def eval_transform(sz=224):
    rs=int(round(sz*256/224)); return transforms.Compose([transforms.Resize(rs),transforms.CenterCrop(sz),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

class CancerImageDataset(Dataset):
    def __init__(self,df,transform): self.df=df.reset_index(drop=True); self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.image_path) as im: x=self.transform(im.convert('RGB'))
        return {'image':x,'target':torch.tensor(int(r.target)),'group_key':str(r.group_key),'image_path':str(r.image_path)}

def subgroup_balanced_sampler(df,seed):
    sizes=df.groupby('group_key').size().to_dict(); w=np.array([1/sizes[g] for g in df.group_key],dtype=np.float64); w/=w.mean()
    gen=torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w,dtype=torch.double),len(df),replacement=True,generator=gen)

def build_model(name,num_classes=3,dropout=.3,pretrained=True):
    n=name.lower()
    if n=='resnet18':
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); f=m.fc.in_features; m.fc=nn.Sequential(nn.Dropout(dropout),nn.Linear(f,num_classes)); return m
    if n in {'efficientnet_b0','efficientnet-b0'}:
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None); f=m.classifier[-1].in_features; m.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(f,num_classes)); return m
    if n in {'mobilenet_v3_large','mobilenetv3_large'}:
        m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None); f=m.classifier[-1].in_features; m.classifier[-1]=nn.Linear(f,num_classes); return m
    raise ValueError(name)

def split_parameter_groups(model,name):
    head=list(model.fc.parameters()) if name.lower()=='resnet18' else list(model.classifier.parameters()); ids={id(p) for p in head}; back=[p for p in model.parameters() if id(p) not in ids]; return back,head

def freeze_backbone(model,name,freeze=True):
    b,h=split_parameter_groups(model,name)
    for p in b: p.requires_grad=not freeze
    for p in h: p.requires_grad=True

@dataclass
class Metrics:
    image_accuracy:float; image_balanced_accuracy:float; image_macro_f1:float; image_weighted_f1:float
    group_accuracy:float; group_balanced_accuracy:float; group_macro_f1:float; group_weighted_f1:float
    image_confusion_matrix:List[List[int]]; group_confusion_matrix:List[List[int]]; class_metrics:Dict[str,Dict[str,float]]

def compute_metrics(targets,probs,groups):
    y=np.asarray(targets,int); yp=probs.argmax(1)
    ia=accuracy_score(y,yp); iba=balanced_accuracy_score(y,yp); im=f1_score(y,yp,average='macro',zero_division=0); iw=f1_score(y,yp,average='weighted',zero_division=0)
    ic=confusion_matrix(y,yp,labels=range(3)).tolist(); gp=defaultdict(list); gt={}
    for t,p,g in zip(y,probs,groups): gp[g].append(p); gt[g]=int(t)
    keys=sorted(gp); P=np.stack([np.mean(gp[k],0) for k in keys]); gy=np.array([gt[k] for k in keys]); gyp=P.argmax(1)
    ga=accuracy_score(gy,gyp); gba=balanced_accuracy_score(gy,gyp); gm=f1_score(gy,gyp,average='macro',zero_division=0); gw=f1_score(gy,gyp,average='weighted',zero_division=0)
    gc=confusion_matrix(gy,gyp,labels=range(3)).tolist(); pr,re,f1,su=precision_recall_fscore_support(gy,gyp,labels=range(3),zero_division=0)
    cm={c:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'support_groups':int(su[i])} for i,c in enumerate(CLASS_ORDER)}
    return Metrics(float(ia),float(iba),float(im),float(iw),float(ga),float(gba),float(gm),float(gw),ic,gc,cm)

def metrics_to_dict(m,prefix=''):
    d={f'{prefix}image_accuracy':m.image_accuracy,f'{prefix}image_balanced_accuracy':m.image_balanced_accuracy,f'{prefix}image_macro_f1':m.image_macro_f1,f'{prefix}image_weighted_f1':m.image_weighted_f1,f'{prefix}group_accuracy':m.group_accuracy,f'{prefix}group_balanced_accuracy':m.group_balanced_accuracy,f'{prefix}group_macro_f1':m.group_macro_f1,f'{prefix}group_weighted_f1':m.group_weighted_f1,f'{prefix}image_confusion_matrix_json':json.dumps(m.image_confusion_matrix),f'{prefix}group_confusion_matrix_json':json.dumps(m.group_confusion_matrix)}
    for c,v in m.class_metrics.items():
        for k,x in v.items(): d[f'{prefix}{c}_{k}']=x
    return d

def make_fixed_group_holdout(df,test_size,seed):
    gl=df.groupby('group_key',as_index=False)['target'].first().sort_values('group_key').reset_index(drop=True); groups=gl.group_key.to_numpy(); labels=gl.target.to_numpy(); n=max(2,int(round(1/test_size)))
    cv=StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed); cand=[]; X=np.zeros(len(gl))
    for tr,te in cv.split(X,labels,groups): cand.append((abs(len(te)/len(gl)-test_size),tr,te))
    _,tr,te=min(cand,key=lambda x:x[0]); trg=set(groups[tr]); teg=set(groups[te])
    return np.where(df.group_key.isin(trg))[0], np.where(df.group_key.isin(teg))[0]

def save_json(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2))

def checkpoint_payload(model,model_name,epoch,best_metric,config):
    return {'model_name':model_name,'state_dict':model.state_dict(),'epoch':int(epoch),'best_group_macro_f1':float(best_metric),'class_order':CLASS_ORDER,'config':config}