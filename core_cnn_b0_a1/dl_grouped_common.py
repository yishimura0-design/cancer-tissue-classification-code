#!/usr/bin/env python3
from __future__ import annotations
import json, math, random, re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms

SCENARIO1_CLASSES={"Normal":["F1","F2","F3","F4","F5","F6","F7","F8"],"Malignant":[f"{g}{i}" for g in "ABCDE" for i in range(1,9)]}
SCENARIO2_CLASSES={"Stage IB":["A2","B5","B6","B8","D1","E1","E8"],"Stage IIA":["A4","A5","A6","B1","B2","B7","C2","C4","C5","C6","C8","D2","D3","D5","D7","D8","E2","E3","E4","E5","E6","E7"],"Stage IIB":["A3","A7","B3","B4","C1","D6"],"Stage III":["A1","C3"],"Stage IV":["A8","C7","D4"],"Normal":["F1","F2","F3","F4","F5","F6","F7","F8"]}
SCENARIOS={"scenario1":SCENARIO1_CLASSES,"scenario2":SCENARIO2_CLASSES}
ANGLE_PATTERN=re.compile(r"(?i)(?:[_-]?ang(?:le)?[_-]?\d+(?:\.\d+)?(?:deg)?(?:[_-]?)?)")

def seed_everything(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=True

def infer_path_column(df):
    for c in ["filepath","file_path","path","image_path","filename"]:
        if c in df.columns:return c
    raise ValueError(f"No image path column found: {list(df.columns)}")

def extract_sample_id(path:str)->str:
    p=Path(path)
    for part in reversed(p.parts):
        if re.fullmatch(r"[A-Fa-f][1-8]",part): return part.upper()
    m=re.search(r"(?<![A-Za-z0-9])([A-Fa-f][1-8])(?![0-9])",p.stem)
    if m:return m.group(1).upper()
    raise ValueError(f"Cannot infer sample ID from {path}")

def make_group_id(path:str)->str:
    p=Path(path); sid=extract_sample_id(path)
    stem=ANGLE_PATTERN.sub("",p.stem); stem=re.sub(r"[_-]+$","",stem)
    return f"{sid[0]}/{sid}/{stem}"

def prepare_dataframe(manifest:str,scenario:str):
    df=pd.read_csv(manifest); pc=infer_path_column(df); df=df.copy(); df["path"]=df[pc].astype(str)
    df=df[df["path"].str.lower().str.endswith((".tif",".tiff"))].copy(); df["sample_id"]=df["path"].map(extract_sample_id)
    lookup={sid:lab for lab,sids in SCENARIOS[scenario].items() for sid in sids}; df["label"]=df["sample_id"].map(lookup); df=df[df["label"].notna()].copy(); df["group_id"]=df["path"].map(make_group_id)
    bad=df.groupby("group_id")["label"].nunique(); bad=bad[bad>1]
    if len(bad): raise RuntimeError(f"Mixed-label groups found: {bad.index[:10].tolist()}")
    class_names=list(SCENARIOS[scenario].keys()); m={n:i for i,n in enumerate(class_names)}; df["target"]=df["label"].map(m).astype(int)
    missing=[p for p in df["path"] if not Path(p).is_file()]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} images; examples: {missing[:5]}")
    return df.reset_index(drop=True),class_names

def make_transforms(train:bool,image_size:int=224):
    norm=transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    if train:return transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(image_size,scale=(0.90,1.0)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(0.10,0.10,0.08,0.02),transforms.ToTensor(),norm])
    return transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),norm])

class CancerImageDataset(Dataset):
    def __init__(self,df,train,subgroup_balance=False,image_size=224):
        self.df=df.reset_index(drop=True).copy(); self.transform=make_transforms(train,image_size)
        counts=self.df["group_id"].value_counts().to_dict(); w=np.array([1.0/counts[g] for g in self.df["group_id"]],dtype=np.float32)
        self.sample_weights=(w/w.mean()) if subgroup_balance else np.ones_like(w)
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r["path"]) as im: x=self.transform(im.convert("RGB"))
        return {"image":x,"target":int(r["target"]),"group_id":str(r["group_id"]),"path":str(r["path"]),"sample_weight":float(self.sample_weights[i])}

def build_model(name,n_classes,pretrained=True):
    n=name.lower()
    if n=="resnet18":
        m=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None); m.fc=nn.Linear(m.fc.in_features,n_classes); m._head_module=m.fc; return m
    if n in {"efficientnet_b0","efficientnet-b0"}:
        m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None); m.classifier[1]=nn.Linear(m.classifier[1].in_features,n_classes); m._head_module=m.classifier; return m
    if n in {"mobilenet_v3_large","mobilenetv3_large","mobilenetv3-large"}:
        m=models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None); m.classifier[3]=nn.Linear(m.classifier[3].in_features,n_classes); m._head_module=m.classifier; return m
    raise ValueError(name)

def set_backbone_frozen(model,frozen):
    for p in model.parameters():p.requires_grad=not frozen
    if frozen:
        for p in model._head_module.parameters():p.requires_grad=True

def moderate_class_weights(train_df,n_classes):
    g=train_df[["group_id","target"]].drop_duplicates("group_id"); counts=g["target"].value_counts().to_dict(); v=np.array([1/math.sqrt(max(counts.get(i,1),1)) for i in range(n_classes)],dtype=np.float32); v/=v.mean(); return torch.tensor(v)

def group_level_frame(df):return df[["group_id","target","label"]].drop_duplicates("group_id").reset_index(drop=True)

def stratified_group_holdout(df,test_size,seed):
    g=group_level_frame(df); a,b=train_test_split(g,test_size=test_size,random_state=seed,stratify=g["target"]); A=df[df["group_id"].isin(set(a["group_id"]))].copy(); B=df[df["group_id"].isin(set(b["group_id"]))].copy(); assert set(A.group_id).isdisjoint(set(B.group_id)); return A.reset_index(drop=True),B.reset_index(drop=True)

def validate_fold_feasibility(df,folds):
    c=group_level_frame(df).groupby("label")["group_id"].nunique().sort_values(); bad=c[c<folds]
    if len(bad):raise ValueError(f"{folds}-fold subgroup CV impossible: {bad.to_dict()}")

def metric_dict(y_true,y_pred,class_names,prefix):
    out={f"{prefix}_acc":float(accuracy_score(y_true,y_pred)),f"{prefix}_macro_f1":float(f1_score(y_true,y_pred,average="macro",zero_division=0)),f"{prefix}_weighted_f1":float(f1_score(y_true,y_pred,average="weighted",zero_division=0))}
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=list(range(len(class_names))),zero_division=0)
    for i,n in enumerate(class_names):
        s=n.replace(" ","_"); out[f"{prefix}_{s}_precision"]=float(p[i]); out[f"{prefix}_{s}_recall"]=float(r[i]); out[f"{prefix}_{s}_f1"]=float(f[i])
    return out

@torch.no_grad()
def evaluate(model,loader,device,class_names,amp_enabled):
    model.eval(); probs=[]; ys=[]; groups=[]; paths=[]
    for b in loader:
        x=b["image"].to(device,non_blocking=True)
        with torch.autocast(device_type=device.type,enabled=amp_enabled): lg=model(x)
        probs.append(torch.softmax(lg,1).cpu().numpy()); ys.extend(b["target"].numpy().tolist()); groups.extend(b["group_id"]); paths.extend(b["path"])
    probs=np.concatenate(probs); y=np.asarray(ys); yp=probs.argmax(1); im=metric_dict(y,yp,class_names,"image")
    gp=defaultdict(list); gt={}
    for g,t,p in zip(groups,y,probs):gp[g].append(p); gt[g]=int(t)
    gids=sorted(gp); gy=np.array([gt[g] for g in gids]); gprob=np.stack([np.mean(gp[g],0) for g in gids]); gyp=gprob.argmax(1); gm=metric_dict(gy,gyp,class_names,"group")
    ip=pd.DataFrame({"path":paths,"group_id":groups,"target":y,"prediction":yp,"target_label":[class_names[i] for i in y],"prediction_label":[class_names[i] for i in yp]})
    for i,n in enumerate(class_names):ip[f"prob_{n}"]=probs[:,i]
    gpdf=pd.DataFrame({"group_id":gids,"target":gy,"prediction":gyp,"target_label":[class_names[i] for i in gy],"prediction_label":[class_names[i] for i in gyp]})
    for i,n in enumerate(class_names):gpdf[f"prob_{n}"]=gprob[:,i]
    return {**im,**gm},ip,gpdf

def save_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2),encoding="utf-8")
