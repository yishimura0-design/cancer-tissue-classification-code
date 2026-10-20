#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 00: Clean dataset into a manifest CSV.

What this does:
  - recursively scans the selected folders
  - uses ONLY .tif/.tiff
  - skips JPEG/PNG automatically because they are not selected
  - skips any folder named ConvertedTIFFs for safety
  - labels images as HPNE, MIA, or PANC from the path
  - keeps same filename from different folders as DIFFERENT samples
  - optionally verifies that images can be opened
  - optionally creates class-folder symlinks without modifying originals

Run:
  python 00_clean_make_manifest.py --out results_angle_cpu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

CLASS_ORDER = ["HPNE", "PANC", "MIA"]
VALID_SUFFIXES = {".tif", ".tiff"}
SKIP_DIR_NAMES = {"ConvertedTIFFs"}

# Users must pass --input-dirs with authorized local image directories.
DEFAULT_INPUT_DIRS = []


def sha1_text(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def path_has_skip_dir(path: Path) -> bool:
    parts_lower = {p.lower() for p in path.parts}
    return any(d.lower() in parts_lower for d in SKIP_DIR_NAMES)


def infer_label(path: Path) -> Optional[str]:
    text = str(path).upper()
    found = []
    # Use token boundaries where possible, but allow names like 5_11_25_HPNE.
    for label in CLASS_ORDER:
        if re.search(rf"(^|[^A-Z0-9]){label}([^A-Z0-9]|$)", text):
            found.append(label)
    if len(found) == 1:
        return found[0]
    return None


def normalize_stem_for_group(stem: str) -> str:
    """
    Remove common angle tokens so same sample at different angles can stay in the
    same group during optional group split. Examples: Ang0, angle_45, _90deg.
    """
    x = stem.lower()
    x = re.sub(r"[_\- ]?ang(le)?[_\- ]?\d+", "", x)
    x = re.sub(r"[_\- ]?\d+[_\- ]?deg", "", x)
    x = re.sub(r"[_\- ]+", "_", x).strip("_-. ")
    return x or stem.lower()


def make_group_key(path: Path, label: str) -> str:
    """
    Group key for optional leakage-safe group splitting.

    Same filename in a different folder is treated as different because parent folder
    is part of the key. Inside the same folder, common angle variants can be grouped
    together by removing Ang0/angle_45/90deg-like tokens from the stem.
    """
    parent = str(path.parent)
    normalized_stem = normalize_stem_for_group(path.stem)
    return sha1_text(f"{label}|{parent}|{normalized_stem}", n=20)


def verify_image(path: Path) -> Tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, str(e)


def iter_files(input_dirs: List[Path]) -> Iterable[Path]:
    for root in input_dirs:
        if not root.exists():
            print(f"WARNING missing input folder: {root}")
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if path_has_skip_dir(p):
                continue
            if p.suffix.lower() in VALID_SUFFIXES:
                yield p


def maybe_symlink(row: Dict[str, str], symlink_root: Path) -> None:
    src = Path(row["path"])
    dst_dir = symlink_root / row["label"]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{row['image_id']}_{src.name}"
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_angle_cpu", help="Output directory")
    ap.add_argument("--input-dirs", nargs="*", default=DEFAULT_INPUT_DIRS, help="Override input folders")
    ap.add_argument("--verify-images", action="store_true", help="Open every image and skip corrupted files")
    ap.add_argument("--make-symlinks", action="store_true", help="Create clean_dataset/CLASS symlinks")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_csv = out / "manifest_clean.csv"
    summary_json = out / "manifest_summary.json"
    duplicate_report_csv = out / "same_filename_different_paths.csv"
    skipped_csv = out / "skipped_files.csv"

    input_dirs = [Path(x) for x in args.input_dirs]

    rows: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    seen_paths = set()

    for path in iter_files(input_dirs):
        abs_path = str(path.resolve())
        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)

        label = infer_label(path)
        if label is None:
            skipped.append({"path": abs_path, "reason": "could_not_infer_unique_label"})
            continue

        if args.verify_images:
            ok, err = verify_image(path)
            if not ok:
                skipped.append({"path": abs_path, "reason": f"cannot_open_image: {err}"})
                continue

        image_id = sha1_text(abs_path, n=20)
        rows.append(
            {
                "image_id": image_id,
                "path": abs_path,
                "label": label,
                "filename": path.name,
                "stem": path.stem,
                "extension": path.suffix.lower(),
                "parent_dir": str(path.parent.resolve()),
                "group_key": make_group_key(path.resolve(), label),
            }
        )

    rows.sort(key=lambda r: (CLASS_ORDER.index(r["label"]), r["parent_dir"], r["filename"]))

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "path", "label", "filename", "stem", "extension", "parent_dir", "group_key"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with open(skipped_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "reason"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(skipped)

    # Same basename from different paths is not removed; this is only a report.
    by_filename: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_filename[r["filename"]].append(r)

    dup_rows: List[Dict[str, str]] = []
    for fname, items in by_filename.items():
        if len(items) > 1:
            for r in items:
                dup_rows.append({
                    "filename": fname,
                    "image_id": r["image_id"],
                    "label": r["label"],
                    "path": r["path"],
                    "note": "kept; same filename in different folder is treated as different sample",
                })

    with open(duplicate_report_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["filename", "image_id", "label", "path", "note"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(dup_rows)

    counts = Counter(r["label"] for r in rows)
    folder_counts = Counter((r["label"], r["parent_dir"]) for r in rows)
    summary = {
        "n_images": len(rows),
        "class_order": CLASS_ORDER,
        "class_counts": {c: counts.get(c, 0) for c in CLASS_ORDER},
        "n_skipped": len(skipped),
        "n_same_filename_report_rows": len(dup_rows),
        "valid_suffixes": sorted(VALID_SUFFIXES),
        "skipped_dir_names": sorted(SKIP_DIR_NAMES),
        "input_dirs": [str(x) for x in input_dirs],
        "manifest_csv": str(manifest_csv),
        "skipped_csv": str(skipped_csv),
        "same_filename_report_csv": str(duplicate_report_csv),
        "folder_counts": [
            {"label": label, "folder": folder, "count": count}
            for (label, folder), count in sorted(folder_counts.items())
        ],
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.make_symlinks:
        symlink_root = out / "clean_dataset_symlinks"
        for r in rows:
            maybe_symlink(r, symlink_root)
        print(f"Symlinks written under: {symlink_root}")

    print("\n===== CLEAN MANIFEST DONE =====")
    print(f"Manifest: {manifest_csv}")
    print(f"Summary : {summary_json}")
    print(f"Skipped : {skipped_csv}")
    print("Class counts:")
    for c in CLASS_ORDER:
        print(f"  {c}: {counts.get(c, 0)}")
    print(f"Total images: {len(rows)}")
    print(f"Skipped files: {len(skipped)}")
    print(f"Same filename report rows kept as different samples: {len(dup_rows)}")

    too_small = [c for c in CLASS_ORDER if counts.get(c, 0) < 5]
    if too_small:
        print(f"WARNING: 5-fold CV needs at least 5 images per class. Too small: {too_small}")


if __name__ == "__main__":
    main()
