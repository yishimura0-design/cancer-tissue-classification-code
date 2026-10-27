#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image loading and feature extraction utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    from skimage.feature import hog, local_binary_pattern  # type: ignore
except Exception:
    hog = None
    local_binary_pattern = None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    resize: Tuple[int, int] = (64, 64)
    stride: int = 1
    grayscale: bool = True
    hist_bins: int = 64
    lbp_points: int = 8
    lbp_radius: int = 1
    hog_pixels_per_cell: int = 16
    hog_cells_per_block: int = 2

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FeatureSpec":
        resize_val = d.get("resize", [64, 64])
        return FeatureSpec(
            name=str(d["name"]),
            kind=str(d["kind"]),
            resize=(int(resize_val[0]), int(resize_val[1])),
            stride=int(d.get("stride", 1)),
            grayscale=bool(d.get("grayscale", True)),
            hist_bins=int(d.get("hist_bins", 64)),
            lbp_points=int(d.get("lbp_points", 8)),
            lbp_radius=int(d.get("lbp_radius", 1)),
            hog_pixels_per_cell=int(d.get("hog_pixels_per_cell", 16)),
            hog_cells_per_block=int(d.get("hog_cells_per_block", 2)),
        )


def load_feature_specs(json_path: str | Path) -> List[FeatureSpec]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [FeatureSpec.from_dict(x) for x in data["search_spaces"]]


def load_image(path: str | Path, resize: Tuple[int, int], grayscale: bool) -> np.ndarray:
    """Load .tif/.tiff safely, fix orientation, resize, return float32 [0,1]."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")
    # BILINEAR is stable and fast for microscopy baseline ML.
    img = img.resize(resize, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def _gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    return arr.mean(axis=2)


def feature_raw(arr: np.ndarray, stride: int = 1) -> np.ndarray:
    if stride > 1:
        arr = arr[::stride, ::stride] if arr.ndim == 2 else arr[::stride, ::stride, :]
    return arr.reshape(-1).astype(np.float32)


def feature_stats(arr: np.ndarray) -> np.ndarray:
    """Simple intensity/channel statistics. Very small dimensional baseline."""
    feats: List[float] = []
    if arr.ndim == 2:
        channels = [arr]
    else:
        channels = [arr[:, :, c] for c in range(arr.shape[2])]
    for ch in channels:
        p = np.percentile(ch, [1, 5, 25, 50, 75, 95, 99])
        feats.extend([
            float(np.mean(ch)),
            float(np.std(ch)),
            float(np.min(ch)),
            float(np.max(ch)),
            *[float(x) for x in p],
        ])
    return np.asarray(feats, dtype=np.float32)


def feature_hist(arr: np.ndarray, bins: int = 64) -> np.ndarray:
    g = _gray(arr)
    hist, _ = np.histogram(g.reshape(-1), bins=bins, range=(0.0, 1.0), density=True)
    return hist.astype(np.float32)


def feature_lbp_hist(arr: np.ndarray, points: int = 8, radius: int = 1) -> np.ndarray:
    if local_binary_pattern is None:
        raise RuntimeError("scikit-image is required for LBP. Install: pip install scikit-image")
    g = (_gray(arr) * 255.0).astype(np.uint8)
    lbp = local_binary_pattern(g, P=points, R=radius, method="uniform")
    n_bins = points + 2
    hist, _ = np.histogram(lbp.reshape(-1), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def feature_hog(arr: np.ndarray, pixels_per_cell: int = 16, cells_per_block: int = 2) -> np.ndarray:
    if hog is None:
        raise RuntimeError("scikit-image is required for HOG. Install: pip install scikit-image")
    g = _gray(arr)
    feat = hog(
        g,
        orientations=9,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def extract_one(path: str | Path, spec: FeatureSpec) -> np.ndarray:
    """
    Feature kinds:
      raw_gray, raw_rgb, hist, stats_hist, lbp, hog, combined_small, combined_hog
    """
    kind = spec.kind
    grayscale = spec.grayscale or kind in {"raw_gray", "hist", "stats_hist", "lbp", "hog", "combined_small", "combined_hog"}
    arr = load_image(path, resize=spec.resize, grayscale=grayscale)

    if kind in {"raw_gray", "raw_rgb"}:
        return feature_raw(arr, stride=spec.stride)
    if kind == "hist":
        return feature_hist(arr, bins=spec.hist_bins)
    if kind == "stats_hist":
        return np.concatenate([feature_stats(arr), feature_hist(arr, bins=spec.hist_bins)]).astype(np.float32)
    if kind == "lbp":
        return feature_lbp_hist(arr, points=spec.lbp_points, radius=spec.lbp_radius)
    if kind == "hog":
        return feature_hog(arr, pixels_per_cell=spec.hog_pixels_per_cell, cells_per_block=spec.hog_cells_per_block)
    if kind == "combined_small":
        return np.concatenate([
            feature_stats(arr),
            feature_hist(arr, bins=spec.hist_bins),
            feature_lbp_hist(arr, points=spec.lbp_points, radius=spec.lbp_radius),
        ]).astype(np.float32)
    if kind == "combined_hog":
        return np.concatenate([
            feature_stats(arr),
            feature_hist(arr, bins=spec.hist_bins),
            feature_lbp_hist(arr, points=spec.lbp_points, radius=spec.lbp_radius),
            feature_hog(arr, pixels_per_cell=spec.hog_pixels_per_cell, cells_per_block=spec.hog_cells_per_block),
        ]).astype(np.float32)

    raise ValueError(f"Unknown feature kind: {kind}")
