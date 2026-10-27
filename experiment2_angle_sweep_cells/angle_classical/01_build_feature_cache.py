#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 01: Build feature matrices for each image search space.

Input:
  results_angle_cpu/manifest_clean.csv
  search_spaces.json

Output:
  results_angle_cpu/features/*.npz

Run:
  python 01_build_feature_cache.py \
    --manifest results_angle_cpu/manifest_clean.csv \
    --search-spaces search_spaces.json \
    --out results_angle_cpu/features
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from feature_lib import FeatureSpec, extract_one, load_feature_specs

CLASS_ORDER = ["HPNE", "PANC", "MIA"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_ORDER)}


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def spec_to_jsonable(spec: FeatureSpec) -> Dict[str, object]:
    return {
        "name": spec.name,
        "kind": spec.kind,
        "resize": list(spec.resize),
        "stride": spec.stride,
        "grayscale": spec.grayscale,
        "hist_bins": spec.hist_bins,
        "lbp_points": spec.lbp_points,
        "lbp_radius": spec.lbp_radius,
        "hog_pixels_per_cell": spec.hog_pixels_per_cell,
        "hog_cells_per_block": spec.hog_cells_per_block,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Clean manifest CSV from Step 00")
    ap.add_argument("--search-spaces", default="search_spaces.json", help="JSON search-space config")
    ap.add_argument("--out", default="results_angle_cpu/features", help="Feature cache output folder")
    ap.add_argument("--only", default="", help="Optional comma list of search-space names to run")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    specs = load_feature_specs(args.search_spaces)
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in specs if s.name in wanted]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_csv = out / "feature_cache_summary.csv"
    failures_csv = out / "feature_failures.csv"

    summary_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, str]] = []

    print(f"Loaded manifest rows: {len(rows)}")
    print(f"Feature spaces to build: {[s.name for s in specs]}")

    for spec in specs:
        t0 = time.time()
        print(f"\n===== Building {spec.name} ({spec.kind}) =====")
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        image_ids: List[str] = []
        paths: List[str] = []
        labels: List[str] = []
        group_keys: List[str] = []

        for i, r in enumerate(rows, start=1):
            if i % 100 == 0 or i == 1 or i == len(rows):
                print(f"  {spec.name}: {i}/{len(rows)}", flush=True)

            label = r["label"]
            if label not in CLASS_TO_ID:
                failure_rows.append({
                    "search_space": spec.name,
                    "image_id": r.get("image_id", ""),
                    "path": r.get("path", ""),
                    "reason": f"unknown_label_{label}",
                })
                continue

            try:
                feat = extract_one(r["path"], spec)
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            except Exception as e:
                failure_rows.append({
                    "search_space": spec.name,
                    "image_id": r.get("image_id", ""),
                    "path": r.get("path", ""),
                    "reason": repr(e),
                })
                continue

            X_list.append(feat)
            y_list.append(CLASS_TO_ID[label])
            image_ids.append(r["image_id"])
            paths.append(r["path"])
            labels.append(label)
            group_keys.append(r.get("group_key", r["image_id"]))

        if not X_list:
            print(f"SKIP {spec.name}: no features built")
            summary_rows.append({
                "search_space": spec.name,
                "status": "failed_no_features",
                "n_samples": 0,
                "dim": 0,
                "seconds": round(time.time() - t0, 2),
                "npz": "",
            })
            continue

        dims = {x.shape[0] for x in X_list}
        if len(dims) != 1:
            raise RuntimeError(f"Feature dimension mismatch for {spec.name}: {sorted(dims)}")

        X = np.vstack(X_list).astype(np.float32)
        y = np.asarray(y_list, dtype=np.int64)
        npz_path = out / f"{spec.name}.npz"
        np.savez_compressed(
            npz_path,
            X=X,
            y=y,
            classes=np.asarray(CLASS_ORDER),
            image_ids=np.asarray(image_ids),
            paths=np.asarray(paths),
            labels=np.asarray(labels),
            group_keys=np.asarray(group_keys),
            spec_json=np.asarray([json.dumps(spec_to_jsonable(spec))]),
        )

        seconds = time.time() - t0
        counts = {c: int(np.sum(y == i)) for i, c in enumerate(CLASS_ORDER)}
        print(f"Saved: {npz_path}")
        print(f"Shape: X={X.shape}, y={y.shape}")
        print(f"Counts: {counts}")
        print(f"Seconds: {seconds:.2f}")

        summary_rows.append({
            "search_space": spec.name,
            "status": "ok",
            "n_samples": X.shape[0],
            "dim": X.shape[1],
            "HPNE": counts["HPNE"],
            "PANC": counts["PANC"],
            "MIA": counts["MIA"],
            "seconds": round(seconds, 2),
            "npz": str(npz_path),
        })

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["search_space", "status", "n_samples", "dim", "HPNE", "PANC", "MIA", "seconds", "npz"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    with open(failures_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["search_space", "image_id", "path", "reason"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(failure_rows)

    print("\n===== FEATURE CACHE DONE =====")
    print(f"Summary : {summary_csv}")
    print(f"Failures: {failures_csv}")
    print(f"Failure rows: {len(failure_rows)}")


if __name__ == "__main__":
    main()
