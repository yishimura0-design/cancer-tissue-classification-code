"""CPU-only safeguards. Run: python -m pytest -q test_angle_cnn_heldout_cv.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import angle_cnn_heldout_cv as runner


def synthetic_manifest():
    # Mirrors audited 939 images, 37 filename-derived independent group keys.
    # HPNE 9x3+4x72=315; PANC 11x3+4x72=321; MIA 5x3+4x72=303.
    rows=[]
    for cl, (num_small,num_large) in enumerate([(9,4),(11,4),(5,4)]):
        for j,n in enumerate([3]*num_small+[72]*num_large):
            k=f'group_{cl}_{j:02}'
            for a in range(n):
                rows.append({'group_key':k,'target':cl,'label':runner.LABELS[cl],
                             'image_path':f'/mock/{k}_angle_{a:03}.tif'})
    return pd.DataFrame(rows)


def test_data_profile_and_split_balance():
    df=synthetic_manifest()
    runner.check_manifest_metadata(df)
    roles=runner.build_role_table(df,seed=42,val_fraction=.2)
    summary=runner.verify_role_table(df,roles)
    assert len(roles)==37*5
    assert len(summary)==5
    assert sorted(x['test_groups'] for x in summary)==[7,7,7,8,8]
    assert all(x['val_groups']>=5 for x in summary)
    assert all(x['train_groups']>=20 for x in summary)
    assert sum(x['test_images'] for x in summary)==939
    assert (roles.role=='test').sum()==37
    assert roles[roles.role=='test'].group_key.nunique()==37
    assert roles.groupby(['fold','role']).target.nunique().eq(3).all()


def test_fold_roles_are_deterministic_but_seed_sensitive():
    df=synthetic_manifest()
    a=runner.build_role_table(df,42)
    b=runner.build_role_table(df,42)
    c=runner.build_role_table(df,43)
    pd.testing.assert_frame_equal(a,b)
    assert not a.equals(c)


def test_subgroup_overlap_tampering_rejected():
    df=synthetic_manifest()
    roles=runner.build_role_table(df)
    modified=roles.copy()
    selected=modified[(modified.fold==1)&(modified.role=='test')].index[0]
    modified.loc[selected,'role']='train'
    with pytest.raises(ValueError,match='coverage|exactly once'):
        runner.verify_role_table(df,modified)


def test_mixed_group_labels_rejected():
    df=synthetic_manifest()
    modified=df.copy()
    modified.loc[0,'target']=1
    with pytest.raises(ValueError,match='multiple class labels'):
        runner.build_role_table(modified)


def test_duplicate_images_rejected():
    df=synthetic_manifest()
    modified=pd.concat([df,df.head(1)],ignore_index=True)
    with pytest.raises(ValueError,match='duplicated image paths'):
        runner.build_role_table(modified)


def test_loss_of_class_rejected():
    df=synthetic_manifest()
    reduced=df[df.target!=2].copy()
    with pytest.raises(ValueError,match='HPNE/PANC/MIA'):
        runner.build_role_table(reduced)


def test_pooled_metric_correctness():
    cm=np.array([[12,1,0],[0,13,2],[0,4,5]])
    r=runner.metrics_from_cm(cm)
    assert r['n']==37
    assert r['accuracy']==pytest.approx(30/37)
    assert r['balanced_accuracy']==pytest.approx((12/13+13/15+5/9)/3)
    assert r['by_class']['MIA']['support']==9


def test_arg_contract():
    a=runner.get_args(['prepare','--manifest','source.csv','--out','new'])
    assert a.inner_val_fraction==.2
    assert a.action=='prepare'
    with pytest.raises(SystemExit):
        runner.get_args(['run'])


def test_reject_existing_or_changed_frozen_plan(tmp_path,monkeypatch):
    # Exercise prepare + frozen role verification with a fake manifest reader.
    df=synthetic_manifest()
    manifest=tmp_path/'manifest.csv'
    manifest.write_text('source dataset v1')
    import dl_common
    monkeypatch.setattr(dl_common,'load_manifest',lambda _:df)
    args=runner.get_args(['prepare','--manifest',str(manifest),'--out',str(tmp_path/'campaign')])
    runner.prepare(args)
    roles,meta=runner.load_frozen_roles(args,df)
    assert meta['n_groups']==37
    assert len(roles)==185
    with pytest.raises(FileExistsError):
        runner.prepare(args)
    manifest.write_text('source dataset v2')
    with pytest.raises(ValueError,match='Manifest hash'):
        runner.load_frozen_roles(args,df)


def test_one_outer_fold_tiny_cnn_end_to_end(tmp_path,monkeypatch):
    """CPU toy CNN smoke: use separate inner validation and outer test images."""
    import torch
    import torch.nn as nn
    from PIL import Image
    import dl_common

    df=synthetic_manifest()
    image_root=tmp_path/'tiny_tiffs'; image_root.mkdir()
    # Small physical TIFFs, varying image intensity. Exactly 939 distinct files.
    new_paths=[]
    for i,r in enumerate(df.itertuples(index=False)):
        p=image_root/f'{i:04}.tif'
        px=np.full((16,16,3),fill_value=(i*7)%255,dtype=np.uint8)
        Image.fromarray(px).save(p)
        new_paths.append(str(p))
    df['image_path']=new_paths
    manifest=tmp_path/'manifest.csv'
    df[['image_path','label','group_key']].rename(columns={'image_path':'path'}).to_csv(manifest,index=False)
    out=tmp_path/'results'
    prep=runner.get_args(['prepare','--manifest',str(manifest),'--out',str(out)])
    runner.prepare(prep)

    class TinyResnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv=nn.Conv2d(3,2,kernel_size=1)
            self.fc=nn.Sequential(nn.Flatten(),nn.Linear(2*8*8,3))
        def forward(self,x):
            return self.fc(self.conv(x))

    monkeypatch.setattr(dl_common,'build_model',lambda *args,**kwargs:TinyResnet())
    run=runner.get_args(['run','--manifest',str(manifest),'--out',str(out),
                        '--model','resnet18','--fold','1','--allow-cpu',
                        '--image-size','8','--max-epochs','1','--min-epochs','1',
                        '--head-epochs','2','--batch-size','128','--num-workers','0'])
    runner.train_one(run)
    fold_dir=out/'models'/'resnet18'/'fold_1'
    metric=json.loads((fold_dir/'fold_metrics.json').read_text())
    assert metric['best_epoch']==1
    assert metric['outer_test_used_for_checkpoint_selection'] is False
    assert metric['early_stopping_metric']=='inner_validation_subgroup_macro_f1'
    assert (fold_dir/'best.pt').is_file()
    test_df=pd.read_csv(fold_dir/'test_group_predictions.csv')
    roles=pd.read_csv(out/'role_assignments.csv')
    assert set(test_df.group_key)==set(roles.loc[(roles.fold==1)&(roles.role=='test'),'group_key'])
    assert set(test_df.group_key).isdisjoint(set(roles.loc[(roles.fold==1)&(roles.role=='val'),'group_key']))
    with pytest.raises(RuntimeError,match='Missing complete heldout fold'):
        runner.summarize(run)