#!/usr/bin/env python3
"""Create a NEW immutable GPU campaign using shared preprocessed PNGs.

Never overwrite the active raw-TIFF pilot or its generated sources/checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from prepare_fullframe_repeat import build, replace_exact, replace_function  # noqa: E402

CORE_TRANSFORMS = '''def make_transforms(train: bool, image_size: int = 224):
    # The cache is generated once with ImageOps.pad(..., BICUBIC, gray 127).
    # Keep spatial preprocessing deterministic and unchanged across all roles.
    def require_cached_size(im):
        if im.size != (image_size, image_size):
            raise ValueError(f'Expected preprocessed {image_size}x{image_size} cache PNG, got {im.size}')
        return im
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base = transforms.Lambda(require_cached_size)
    if train:
        return transforms.Compose([base,
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.10, 0.10, 0.08, 0.02),
            transforms.ToTensor(), norm])
    return transforms.Compose([base, transforms.ToTensor(), norm])
'''

ANGLE_TRAIN = '''def train_transform(sz=224):
    def require_cached_size(im):
        if im.size != (sz, sz):
            raise ValueError(f'Expected {sz}x{sz} cache PNG, got {im.size}')
        return im
    return transforms.Compose([transforms.Lambda(require_cached_size),
        transforms.ColorJitter(brightness=.1, contrast=.1),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
'''

ANGLE_EVAL = '''def eval_transform(sz=224):
    def require_cached_size(im):
        if im.size != (sz, sz):
            raise ValueError(f'Expected {sz}x{sz} cache PNG, got {im.size}')
        return im
    return transforms.Compose([transforms.Lambda(require_cached_size),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
'''


def apply(path: Path, change):
    text = path.read_text()
    fixed = change(text)
    compile(fixed, str(path), 'exec')
    path.write_text(fixed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.repo.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing existing campaign directory: {out}')
    out.mkdir(parents=True)
    try:
        if build(root, out) != 57:
            raise RuntimeError('Unexpected job count')
        core = out/'generated/core/dl_grouped_common.py'
        def core_patch(text):
            text = replace_function(text, 'make_transforms', CORE_TRANSFORMS)
            return replace_exact(text,
                'with Image.open(r["path"]) as im: x=self.transform(im.convert("RGB"))',
                'with Image.open(r["cache_path"]) as im: x=self.transform(im.convert("RGB"))',
                'core cache loader')
        apply(core, core_patch)
        angle = out/'generated/angle/dl_common.py'
        def angle_patch(text):
            text = replace_function(text, 'train_transform', ANGLE_TRAIN)
            text = replace_function(text, 'eval_transform', ANGLE_EVAL)
            return replace_exact(text,
                "with Image.open(r.image_path) as im: x=self.transform(im.convert('RGB'))",
                "with Image.open(r.cache_path) as im: x=self.transform(im.convert('RGB'))",
                'angle cache loader')
        apply(angle, angle_patch)
        for src, dst in [
            ('experiment1_tissue/sensitivity_fullframe/pbs_prepare_cached.pbs', 'pbs_prepare_cached.pbs'),
            ('experiment1_tissue/sensitivity_fullframe/pbs_train_cached.pbs', 'pbs_train_cached.pbs'),
            ('experiment1_tissue/sensitivity_fullframe/submit_cached_campaign.sh','submit_cached_campaign.sh'),
            ('experiment1_tissue/sensitivity_fullframe/prepare_shared_cache.py','prepare_shared_cache.py'),
            ('experiment1_tissue/sensitivity_fullframe/summarize_fullframe_repeat.py','summarize_fullframe_repeat.py'),
        ]:
            shutil.copyfile(root/src, out/dst)
        # The original preflight checks roles, local pretrained weights and 57 jobs.
        # Adapt only its new campaign directory, never the historical experiment.
        preflight = (root/'experiment1_tissue/sensitivity_fullframe/preflight_fullframe_repeat.py').read_text()
        preflight = replace_exact(preflight, "campaign=root/'cancer_fullframe_3cnn_v1'",
                                  "campaign=root/'cancer_fullframe_cached_v1'", 'preflight campaign')
        (out/'preflight_fullframe_cached.py').write_text(preflight)
        manifest = json.loads((out/'build_manifest.json').read_text())
        manifest['experiment'] = 'FULLFRAME_CACHED_V1'
        manifest['preprocess'] = ('Cache original TIFF once to lossless 224x224 RGB PNG with '
                                  'aspect-preserving ImageOps.pad BICUBIC gray127; '
                                  'training flips/ColorJitter on cached pixels, val/test no augmentation')
        manifest['raw_image_paths_preserved_for_labels_groups_and_reporting'] = True
        manifest['new_result_isolation'] = str(out)
        manifest['code_hashes'] = [
            {'file':str(p.relative_to(out)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted((out/'generated').rglob('*.py'))]
        (out/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        for source in (out/'generated').rglob('*.py'):
            compile(source.read_text(), str(source), 'exec')
        assert len((out/'jobs.tsv').read_text().splitlines()) == 57
        print(f'CACHED_CAMPAIGN_BUILD_PASS: 57 jobs, immutable new root {out}')
    except BaseException:
        # Avoid leaving a half-built campaign directory that may later be submitted.
        shutil.rmtree(out)
        raise


if __name__ == '__main__':
    main()
