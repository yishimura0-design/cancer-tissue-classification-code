import random
import numpy as np
import pytest
import torch
from torch import nn
from cnn_checkpoint_state import (rng_snapshot,restore_rng,atomic_checkpoint,
                                  load_checkpoint,snapshot_state)


def new_train():
    m=nn.Sequential(nn.Linear(4,6),nn.ReLU(),nn.Dropout(.2),nn.Linear(6,2))
    opt=torch.optim.AdamW(m.parameters(),lr=.005)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min')
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    return m,opt,scheduler,scaler


def step(model,opt,scheduler,epoch):
    rng = torch.Generator().manual_seed(42+epoch*100003)
    x=torch.randn(16,4,generator=rng)
    y=torch.arange(16)%2
    model.train()
    opt.zero_grad()
    loss=nn.functional.cross_entropy(model(x),y)
    loss.backward();opt.step();scheduler.step(loss.item())
    return float(loss.item())


def test_atomic_checkpoint_restores_full_deterministic_progress(tmp_path):
    identity={'manifest':'frozen', 'config':{'fold':2}}
    random.seed(4);np.random.seed(4);torch.manual_seed(4)
    m,o,sch,scaler=new_train()
    history=[]
    for ep in range(1,5):
        history.append({'epoch':ep,'loss':step(m,o,sch,ep)})
        if ep==2:
            state=snapshot_state(ep,identity,m,o,sch,scaler,history.copy(),
                        {k:v.clone() for k,v in m.state_dict().items()},1.0,1,7,True)
            atomic_checkpoint(tmp_path/'fold/checkpoint_latest.pt',state)
            reference={k:v.clone() for k,v in m.state_dict().items()}
    uninterrupted={k:v.clone() for k,v in m.state_dict().items()}
    # Simulate a restarted Python process: fresh net, fresh optimizer and new RNG.
    random.seed(999);np.random.seed(999);torch.manual_seed(999)
    m2,o2,s2,g2=new_train()
    checkpoint=load_checkpoint(tmp_path/'fold/checkpoint_latest.pt',identity)
    assert checkpoint['completed_epoch']==2
    assert checkpoint['best_epoch']==1
    assert checkpoint['patience_remaining']==7
    assert checkpoint['unfrozen']
    for k,v in reference.items():
        assert torch.equal(checkpoint['model'][k],v)
    m2.load_state_dict(checkpoint['model'])
    o2.load_state_dict(checkpoint['optimizer'])
    s2.load_state_dict(checkpoint['scheduler'])
    g2.load_state_dict(checkpoint['scaler'])
    restore_rng(checkpoint['rng'])
    for ep in range(3,5):
        step(m2,o2,s2,ep)
    for name,weights in uninterrupted.items():
        assert torch.equal(weights,m2.state_dict()[name]),name


def test_checkpoint_wrong_data_or_code_fails_closed(tmp_path):
    torch.manual_seed(4)
    m,o,s,g=new_train()
    p=tmp_path/'ckpt.pt'
    atomic_checkpoint(p,snapshot_state(1,{'code_sha':'original'},m,o,s,g,
                                     [{'epoch':1}],m.state_dict(),1.,1,3,False))
    with pytest.raises(RuntimeError,match='fingerprint mismatch'):
        load_checkpoint(p,{'code_sha':'modified'})
    assert load_checkpoint(p,{'code_sha':'original'})['completed_epoch']==1


def test_atomic_save_never_leaves_partial_temp(tmp_path):
    p=tmp_path/'state.pt'
    atomic_checkpoint(p,{'format_version':1,'run_contract':{},'completed_epoch':1})
    initial=p.read_bytes()
    class Exploding:
        def __reduce__(self):
            raise RuntimeError('write failed')
    with pytest.raises(RuntimeError,match='write failed'):
        atomic_checkpoint(p,{'object':Exploding()})
    assert p.read_bytes()==initial
    assert not list(tmp_path.glob('.checkpoint.*.tmp'))


def test_rng_python_numpy_torch_restored():
    random.seed(16);np.random.seed(16);torch.manual_seed(16)
    state=rng_snapshot()
    expected=(random.random(),float(np.random.rand()),float(torch.rand(())))
    random.random();np.random.rand();torch.rand(())
    restore_rng(state)
    actual=(random.random(),float(np.random.rand()),float(torch.rand(())))
    assert expected==actual
