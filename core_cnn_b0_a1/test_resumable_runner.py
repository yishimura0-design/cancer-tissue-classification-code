"""CPU end-to-end interruption/resume integration with synthetic 45-Core splits."""
import json
import sys
import pandas as pd
import pytest
import torch
from torch import nn
import run_one_core_cnn_fold_resumable as runner


class SyntheticImageDataset(torch.utils.data.Dataset):
    def __init__(self, frame, train, subgroup_balance, image_size):
        self.rows=frame.reset_index(drop=True)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        # No sensitive real images, no GPU required.
        image=torch.tensor([i/100, i/50, i/20, 1.],dtype=torch.float32)
        return {'image':image,'target':int(r.target),'sample_weight':1.,
                'group_id':str(r.group_id),'path':str(r.path)}


class TinyNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.fc=nn.Linear(4,n)
        self.drop=nn.Dropout(.2)
        self._head_module=self.fc
    def forward(self,x):return self.fc(self.drop(x))


@pytest.fixture
def prepare(monkeypatch,tmp_path):
    labels=['Stage IB','Stage IIA','Stage IIB','Stage III','Stage IV','Normal']
    cids=[f'C{i}' for i in range(45)]
    roles=pd.DataFrame({'core_id':cids,'outer_fold':[1]*45,
                        'role':['train']*29+['val']*7+['test']*9})
    # all six labels represented in train, none Stage III/IV in val
    ls=[labels[i%6] for i in range(29)]+[labels[i%3] for i in range(7)]+[labels[i%6] for i in range(9)]
    df=pd.DataFrame([{'sample_id':cid,'path':f'synth_{i}_{f}.tif',
                      'group_id':f'{cid}/fov_{f}','label':ls[i],
                      'target':labels.index(ls[i])}
                     for i,cid in enumerate(cids) for f in range(5)])
    manifest=tmp_path/'manifest.csv';manifest.write_text('fake manifest')
    rolefile=tmp_path/'roles.csv';roles.to_csv(rolefile,index=False)
    monkeypatch.setattr(runner,'prepare_dataframe',lambda a,b:(df.copy(),labels))
    monkeypatch.setattr(runner,'load_roles',lambda a,b:roles)
    monkeypatch.setattr(runner,'CancerImageDataset',SyntheticImageDataset)
    monkeypatch.setattr(runner,'build_model',lambda name,n,pretrained:TinyNet(n))
    return manifest,rolefile


def run(monkeypatch,manifest,roles,out,yield_after):
    args=['runner','--manifest',str(manifest),'--roles',str(roles),
          '--scenario','scenario2','--model','resnet18','--fold','1',
          '--out',str(out),'--epochs','3','--patience','8','--freeze-epochs','1',
          '--batch-size','4','--workers','0','--no-amp','--yield-after-epochs',str(yield_after)]
    monkeypatch.setattr(sys,'argv',args)
    return runner.main()


def test_runner_yields_then_resume_and_protects_complete_fold(monkeypatch,tmp_path,prepare):
    manifest,roles=prepare
    root=tmp_path/'work'
    with pytest.raises(SystemExit) as ex:
        run(monkeypatch,manifest,roles,root,1)
    assert ex.value.code==75
    folder=root/'scenario2/resnet18/fold_1'
    assert (folder/'checkpoint_latest.pt').stat().st_size>0
    assert not (folder/'fold_metrics.json').exists()
    # Run to completion with the same data/config/weights code fingerprint.
    run(monkeypatch,manifest,roles,root,0)
    metrics=json.loads((folder/'fold_metrics.json').read_text())
    assert metrics['epochs_ran']==3
    assert metrics['outer_test_used_for_checkpoint_selection'] is False
    assert metrics['outer_test_image_count']==45
    assert len(pd.read_csv(folder/'test_core_predictions.csv'))==9
    assert len(pd.read_csv(folder/'test_image_predictions.csv.gz'))==45
    # Next run recognizes a complete fold rather than retraining.
    run(monkeypatch,manifest,roles,root,0)


def test_run_contract_changes_are_rejected(monkeypatch,tmp_path,prepare):
    manifest,roles=prepare
    root=tmp_path/'work'
    with pytest.raises(SystemExit):run(monkeypatch,manifest,roles,root,1)
    manifest.write_text('changed manifest contents')
    with pytest.raises(RuntimeError,match='fingerprint'):
        run(monkeypatch,manifest,roles,root,0)
