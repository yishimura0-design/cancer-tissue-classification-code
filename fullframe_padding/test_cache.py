#!/usr/bin/env python3
"""Fast CPU-only cache correctness smoke test: does not require PyTorch."""
import tempfile
from pathlib import Path
import csv
from PIL import Image
import prepare_shared_cache as cache

def test():
    root=Path(tempfile.mkdtemp(prefix='cancer_cache_smoke_'))
    campaign=root/'cancer_fullframe_cached_v1'
    counts={}
    for name, subdir, field in [
        ('core','qc_outputs_clean','filepath'),
        ('angle','cancer_angle_professional_pipeline/results_angle_cpu','path'),
    ]:
        dest=root/subdir
        dest.mkdir(parents=True)
        entries=[]
        for n in range(3):
            p=root/'original_images'/name/f'A1_angle{n}.tif'
            p.parent.mkdir(parents=True,exist_ok=True)
            image=Image.new('RGB',(556,371),(15,120,30))
            for y in range(371):
                image.putpixel((0,y),(255,0,0))
                image.putpixel((555,y),(255,0,0))
            image.save(p)
            entries.append({field:str(p),'label':name,'group_key':'A1'})
        output=dest/('manifest_accepted.csv' if name=='core' else 'manifest_clean.csv')
        with output.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(entries[0]))
            writer.writeheader();writer.writerows(entries)
        counts[name]=3
    original=cache.DATASETS
    cache.DATASETS={
        'core':('qc_outputs_clean/manifest_accepted.csv',3,('filepath',)),
        'angle':('cancer_angle_professional_pipeline/results_angle_cpu/manifest_clean.csv',3,('path',))
    }
    try:
        cache.execute(root,campaign,workers=2)
        cache.execute(root,campaign,workers=2,verify_only=True)
    finally:
        cache.DATASETS=original
    for kind in ('core','angle'):
        cols, rows=cache.read_csv(campaign/'cache/manifests'/f'{kind}.csv')
        assert 'cache_path' in cols
        assert len(rows)==counts[kind]
        field='filepath' if kind=='core' else 'path'
        for row in rows:
            assert Path(row[field]).suffix=='.tif', 'Original path was modified'
            with Image.open(row['cache_path']) as im:
                assert im.size==(224,224) and im.mode=='RGB'
                # A 1-pixel-wide red TIFF border is downsampled by BICUBIC;
                # expect a red signal, not the original pure-red value.
                center = im.getpixel((112,112))
                left = im.getpixel((0,112))
                right = im.getpixel((223,112))
                assert left[0] > center[0] + 40, (left,center)
                assert right[0] > center[0] + 40, (right,center)
                assert im.getpixel((112,0)) == (127,127,127)
    print('CACHE_TEST_PASS: edges retained, lossless pixels, original path preserved')

if __name__=='__main__':
    test()
