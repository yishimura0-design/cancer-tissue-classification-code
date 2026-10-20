import tempfile
from pathlib import Path
from PIL import Image
import torch
from torchvision import models,transforms
from torch import nn
import importlib.util

from prepare_fullframe_repeat import (build,CORE_TRANSFORMS,CORE_MODELS,ANGLE_TRAIN,ANGLE_EVAL,ANGLE_MODELS,ANGLE_SPLIT,MODELS)


def mock_source():
    root=Path(tempfile.mkdtemp(prefix='src_full_'))
    (root/'angle_cnn_heldout_v2').mkdir()
    for file in ('cnn_checkpoint_state.py','core_eval.py','run_contract.py'):
        (root/file).write_text('value = 3\n')
    (root/'dl_grouped_common.py').write_text('def make_transforms(train, image_size=224):\n    return None\n\ndef build_model(name,n_classes,pretrained=True):\n    return None\n')
    (root/'run_one_core_cnn_fold_resumable.py').write_text("choices=['resnet18','efficientnet_b0','mobilenet_v3_large']\n")
    (root/'angle_cnn_heldout_v2/dl_common.py').write_text('def train_transform(sz=224):\n    return None\n\ndef eval_transform(sz=224):\n    return None\n\ndef build_model(name,num_classes=3,dropout=.3,pretrained=True):\n    return None\n\ndef split_parameter_groups(model,name):\n    return None\n')
    (root/'angle_cnn_heldout_v2/angle_cnn_heldout_cv.py').write_text("MODELS = ('resnet18', 'efficientnet_b0', 'mobilenet_v3_large')\n")
    return root


def test_generator():
    root=mock_source();out=Path(tempfile.mkdtemp(prefix='dest_full_'))
    assert build(root,out)==57
    jobs=(out/'jobs.tsv').read_text().splitlines()
    assert len(jobs)==57
    assert len([j for j in jobs if j.startswith('core\tB0\t')])==30
    assert len([j for j in jobs if j.startswith('core\tA1\t')])==12
    assert len([j for j in jobs if j.startswith('angle\tANGLE\t')])==15
    assert all(j.split('\t')[-1] in ('2','4') for j in jobs if j.startswith('core\tA1\t'))
    assert len((out/'build_manifest.json').read_text())>100
    # Generated modules are compilable, without altering source tree.
    for file in (out/'generated').rglob('*.py'):
        compile(file.read_text(),str(file),'exec')
    assert 'RandomResizedCrop' not in (out/'generated/core/dl_grouped_common.py').read_text()
    assert 'RandomResizedCrop' not in (out/'generated/angle/dl_common.py').read_text()
    print('GENERATOR PASS',len(jobs),'jobs, 7 source files')


def test_input_transform():
    # 3:2 source with bright red left and right edge; check that new pad keeps both.
    img=Image.new('RGB',(556,371),(12,90,40))
    for x in list(range(0,20))+list(range(536,556)):
        for y in range(371): img.putpixel((x,y),(255,0,0))
    env={'Image':Image,'transforms':transforms,'models':models,'nn':nn}
    exec(CORE_TRANSFORMS,env)
    f=env['make_transforms']
    eval_t=f(False,224)(img)
    train_t=f(True,224)(img)
    assert eval_t.shape==(3,224,224) and train_t.shape==(3,224,224)
    assert eval_t[0,112,2].item()>1.5 and eval_t[0,112,-3].item()>1.5
    # Padding leaves a band above and below without disturbing aspect ratio.
    assert torch.allclose(eval_t[:,0,0],eval_t[:,0,200])
    exec(ANGLE_TRAIN+ANGLE_EVAL,env)
    assert env['eval_transform'](224)(img).shape==(3,224,224)
    assert env['train_transform'](224)(img).shape==(3,224,224)
    print('TRANSFORM PASS: both retain red edge strips')


def test_model_registry():
    env={'models':models,'nn':nn}
    exec(CORE_MODELS,env)
    for name in MODELS:
        m=env['build_model'](name,3,False)
        p=sum(x.numel() for x in m.parameters())
        assert p>1000000 and hasattr(m,'_head_module'),(name,p)
        # Do not run large forward on CPU here; graph compile is sufficient.
        del m
    env={'models':models,'nn':nn}
    # These are separate generated function definitions. Executing them
    # independently avoids an invalid `raise ...def` token boundary.
    exec(ANGLE_MODELS,env)
    exec(ANGLE_SPLIT,env)
    for name in MODELS:
        m=env['build_model'](name,3,.3,False)
        body,head=env['split_parameter_groups'](m,name)
        assert len(body)>0 and len(head)>0
        assert len(set(map(id,body))&set(map(id,head)))==0
        assert sum(x.numel() for x in body+head)==sum(x.numel() for x in m.parameters())
        del m
    print('ARCHITECTURE PASS: three model heads and two-stage fine-tune parameter partitions')

if __name__=='__main__':
    test_generator();test_input_transform();test_model_registry()
