#!/usr/bin/env python3
"""One shared, lossless-after-resize 224x224 PNG cache for all 57 CNN jobs.

Original TIFF path columns, core identifiers and Angle subgroup memberships
are NEVER replaced. Instead an additional cache_path column supplies the
training/evaluation image loader. This is resumable and checks every PNG.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageOps

SIZE = (224, 224)
FILL = (127, 127, 127)
VERSION = 'full-frame-pad-bicubic-gray127-rgb-224-v1'
DATASETS = {
    'core': ('qc_outputs_clean/manifest_accepted.csv', 16200,
             ('filepath','file_path','path','image_path','filename')),
    'angle': ('cancer_angle_professional_pipeline/results_angle_cpu/manifest_clean.csv', 939,
              ('path','filepath','file_path','image_path')),
}


def read_csv(path):
    with path.open(newline='') as f:
        rdr = csv.DictReader(f)
        cols = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not cols or not rows:
        raise ValueError(f'Empty CSV: {path}')
    return cols, rows


def choose_path_column(cols, precedence):
    for col in precedence:
        if col in cols:
            return col
    raise ValueError(f'Image path column absent: {cols}')


def signature(source: Path) -> str:
    s = source.stat()
    key = json.dumps([VERSION, str(source.resolve()), s.st_size, s.st_mtime_ns],
                     separators=(',', ':'))
    return hashlib.sha256(key.encode()).hexdigest()


def cache_path(source: Path, cache_root: Path) -> Path:
    key = signature(source)
    return cache_root/'images'/key[:2]/f'{key}.png'


def create_png(pair):
    src, dst = pair
    if dst.is_file():
        try:
            with Image.open(dst) as im:
                if im.mode == 'RGB' and im.size == SIZE:
                    im.verify()
                    return False
        except (OSError, ValueError):
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as original:
        frame = ImageOps.pad(original.convert('RGB'), SIZE,
                             method=Image.Resampling.BICUBIC,
                             color=FILL, centering=(0.5,0.5))
    fd, tmp = tempfile.mkstemp(prefix='.building-', suffix='.png', dir=dst.parent)
    os.close(fd)
    try:
        frame.save(tmp, format='PNG', compress_level=1)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True


def validate_exact(src: Path, cached: Path):
    with Image.open(src) as im:
        expected = ImageOps.pad(im.convert('RGB'), SIZE,
                                method=Image.Resampling.BICUBIC,
                                color=FILL, centering=(0.5,0.5))
    with Image.open(cached) as actual:
        if actual.size != SIZE or actual.mode != 'RGB' or expected.tobytes() != actual.tobytes():
            raise AssertionError(f'Pixel mismatch between TIFF and cache: {src}')


def write_manifest(dest: Path, fields, rows):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.manifest-', suffix='.csv',dir=dest.parent)
    os.close(fd)
    try:
        with open(tmp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        if dest.is_file():
            if dest.read_bytes() != Path(tmp).read_bytes():
                raise RuntimeError(f'Existing manifest differs; refuse changing active campaign: {dest}')
        else:
            os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def execute(root: Path, campaign: Path, workers: int, verify_only=False):
    if workers < 1 or workers > 64:
        raise ValueError('workers must be between 1 and 64')
    caches = campaign/'cache'
    all_pairs = {}
    manifest_specs = []
    for name,(rel,count,priority) in DATASETS.items():
        origin = root/rel
        fields, rows = read_csv(origin)
        if len(rows) != count:
            raise RuntimeError(f'{name}: expected {count} original image rows, got {len(rows)}')
        col = choose_path_column(fields, priority)
        if 'cache_path' in fields:
            raise RuntimeError(f'{name}: unexpected preexisting cache_path in original CSV')
        updated = []
        for row in rows:
            source = Path(row[col])
            if not source.is_file() or source.suffix.lower() not in ('.tif','.tiff'):
                raise FileNotFoundError(f'Original TIFF missing/invalid: {source}')
            dest = cache_path(source, caches)
            all_pairs[str(source)] = (source,dest)
            updated.append(dict(row,cache_path=str(dest)))
        manifest_specs.append((name,fields+['cache_path'],updated))
    pairs = list(all_pairs.values())
    print(f'CACHE_PLAN: {len(pairs)} unique original TIFFs across 2 manifests',flush=True)
    if not verify_only:
        new=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n,created in enumerate(pool.map(create_png,pairs),1):
                new += int(created)
                if n%500==0 or n==len(pairs):
                    print(f'CACHE_PROGRESS: {n}/{len(pairs)}; newly written {new}',flush=True)
        print(f'CACHE_WRITE_COMPLETE: {new} created, {len(pairs)-new} reused',flush=True)
    for name, cols, rows in manifest_specs:
        path = caches/'manifests'/f'{name}.csv'
        if not verify_only:
            write_manifest(path, cols, rows)
        saved_cols, saved_rows = read_csv(path)
        if saved_cols != cols or saved_rows != rows:
            raise RuntimeError(f'Manifest mismatch: {path}')
        for row in saved_rows:
            c = Path(row['cache_path'])
            if not c.is_file():
                raise FileNotFoundError(f'Cached PNG not found: {c}')
        for row in (rows[0], rows[len(rows)//2], rows[-1]):
            column = choose_path_column(cols, DATASETS[name][2])
            validate_exact(Path(row[column]), Path(row['cache_path']))
        print(f'CACHE_DATASET_PASS: {name}, {len(rows)} unchanged labels/paths, '
              f'3 original-vs-cache exact-pixel comparisons',flush=True)
    print(f'CACHE_VALIDATION_PASS: all {sum(len(rows) for _,_,rows in manifest_specs)} rows mapped; originals untouched',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('./private_data'))
    p.add_argument('--campaign',type=Path)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    execute(a.root,a.campaign or a.root/'cancer_fullframe_cached_v1',a.workers,a.verify_only)
