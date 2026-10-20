#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean cancer-tissue classical ML search with group-aware five-fold CV.

This is the canonical paper script. It intentionally does not perform a 70/30
holdout or fit a final deployment model. It performs internal validation only:

- Scenario 1: Normal vs Malignant
- Scenario 2: Stage IB/IIA/IIB/III/IV/Normal
- Leakage-safe subgroup splitting: all angular images from one subgroup stay in
  the same fold.
- One fixed fold assignment per scenario, reused by every feature/model.
- Image-level and subgroup-level metrics.
- Out-of-fold (OOF) image and subgroup predictions.
- Fold metrics, pooled OOF metrics, per-class metrics, and confusion matrices.
- Explicit skip reasons for impractical configurations.

Best configurations are selected by mean subgroup-level Macro-F1 across five
folds. The subgroup prediction averages probabilities or decision scores when
available; otherwise it uses majority vote.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import re
import sys
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import sklearn
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from skimage.feature import hog as skimage_hog
    import skimage
except Exception:
    skimage_hog = None
    skimage = None


Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


SCENARIO1_CLASSES = {
    "Normal": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
    "Malignant": [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
    ],
}

SCENARIO2_CLASSES = {
    "Stage IB": ["A2", "B5", "B6", "B8", "D1", "E1", "E8"],
    "Stage IIA": [
        "A4", "A5", "A6", "B1", "B2", "B7",
        "C2", "C4", "C5", "C6", "C8",
        "D2", "D3", "D5", "D7", "D8",
        "E2", "E3", "E4", "E5", "E6", "E7",
    ],
    "Stage IIB": ["A3", "A7", "B3", "B4", "C1", "D6"],
    "Stage III": ["A1", "C3"],
    "Stage IV": ["A8", "C7", "D4"],
    "Normal": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
}

SCENARIOS = {
    "scenario1": {
        "label_col": "scenario1_label",
        "classes": ["Normal", "Malignant"],
    },
    "scenario2": {
        "label_col": "scenario2_label",
        "classes": ["Stage IB", "Stage IIA", "Stage IIB", "Stage III", "Stage IV", "Normal"],
    },
}

EXPECTED_CLEAN_PROFILE = {
    "n_images": 16200,
    "n_subgroups": 225,
    "group_size": 72,
    "n_sample_ids": 45,
}

ANGLE_TOKEN_AT_END_RE = re.compile(
    r"(?i)(?:[_\-\s]*)(?:ang|angle)[_\-\s]*[+-]?\d+(?:\.\d+)?$"
)


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "unnamed"


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def mean_std(values: Iterable[Any]) -> Tuple[float, float]:
    vals = [finite_float(value) for value in values]
    vals = [value for value in vals if np.isfinite(value)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def get_model_classes(model: Any) -> List[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and isinstance(model, Pipeline):
        classes = getattr(model.named_steps.get("clf"), "classes_", None)
    return [str(value) for value in classes] if classes is not None else []


def environment_information() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": getattr(Image, "__version__", "unknown"),
        "scikit_learn": sklearn.__version__,
        "scikit_image": getattr(skimage, "__version__", None),
        "lightgbm": getattr(lgb, "__version__", None),
        "joblib": joblib.__version__,
    }


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def parse_models(models_arg: str) -> List[str]:
    if models_arg.strip().lower() in {"all", "default"}:
        return [
            "ExtraTrees",
            "RandomForest",
            "LinearSVC",
            "SGD_LogLoss",
            "LogisticRegression",
            "DecisionTree",
            "GradientBoosting",
            "GaussianNB",
            "KNN",
            "SVC_RBF",
            "SklearnMLP",
            "LightGBM",
        ]
    if models_arg.strip().lower() == "fast":
        return ["ExtraTrees", "RandomForest", "LinearSVC", "SGD_LogLoss", "DecisionTree"]
    return [name.strip() for name in models_arg.split(",") if name.strip()]


def should_skip_model(model_name: str, n_train: int, dim: int) -> Optional[str]:
    """Skip only configurations that are clearly impractical."""
    if model_name == "KNN" and dim > 20_000:
        return (
            "not completed because of computational and memory limitations: "
            f"KNN with n_features={dim}"
        )
    if model_name == "SVC_RBF" and dim > 5_000:
        return (
            "not completed because of computational and memory limitations: "
            f"RBF-SVC with n_features={dim}"
        )
    if model_name in {"LogisticRegression", "SklearnMLP"} and dim > 500_000:
        return (
            "not completed because of computational and memory limitations: "
            f"{model_name} with n_features={dim}"
        )
    if model_name == "GradientBoosting" and dim > 300_000:
        return (
            "not completed because of computational and memory limitations: "
            f"GradientBoosting with n_features={dim}"
        )
    if model_name == "GaussianNB" and dim > 1_000_000:
        return (
            "not completed because of memory limitations: "
            f"GaussianNB with n_features={dim}"
        )
    if model_name == "LightGBM" and lgb is None:
        return "LightGBM package was unavailable"
    return None


def build_base_model(
    model_name: str,
    seed: int,
    n_estimators: int,
    max_iter: int,
) -> Any:
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_name == "LinearSVC":
        return LinearSVC(
            C=1.0,
            max_iter=max_iter,
            random_state=seed,
            class_weight="balanced",
        )
    if model_name == "SGD_LogLoss":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.15,
            alpha=1e-4,
            max_iter=max_iter,
            random_state=seed,
            class_weight="balanced",
        )
    if model_name == "LogisticRegression":
        return LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=max_iter,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        )
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed, class_weight="balanced")
    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=max(50, n_estimators // 3),
            random_state=seed,
        )
    if model_name == "GaussianNB":
        return GaussianNB()
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if model_name == "SVC_RBF":
        return SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
    if model_name == "SklearnMLP":
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=max(80, max_iter // 10),
            random_state=seed,
        )
    if model_name == "LightGBM":
        if lgb is None:
            raise RuntimeError("LightGBM is unavailable")
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            verbosity=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def needs_scaler(model_name: str) -> bool:
    return model_name in {
        "LinearSVC",
        "SGD_LogLoss",
        "LogisticRegression",
        "KNN",
        "SVC_RBF",
        "SklearnMLP",
    }


def build_model_pipeline(
    model_name: str,
    seed: int,
    n_estimators: int,
    max_iter: int,
    dim: int,
    scale_dim_limit: int,
) -> Tuple[Any, bool]:
    base = build_base_model(model_name, seed, n_estimators, max_iter)
    use_scaler = needs_scaler(model_name) and dim <= scale_dim_limit
    if use_scaler:
        return Pipeline([("scaler", StandardScaler()), ("clf", base)]), True
    return base, False


def estimator_accepts_sample_weight(model: Any) -> bool:
    target = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    try:
        signature = inspect.signature(target.fit)
    except (TypeError, ValueError):
        return False
    return "sample_weight" in signature.parameters


def fit_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: Sequence[str],
) -> Dict[str, Any]:
    """Fit and accurately record whether subgroup-size weights were applied."""
    group_counts = Counter(map(str, groups_train))
    raw_weights = np.asarray(
        [1.0 / group_counts[str(group)] for group in groups_train], dtype=np.float64
    )

    if np.allclose(raw_weights, raw_weights[0]):
        model.fit(X_train, y_train)
        return {
            "sample_weight_used": 0,
            "sample_weight_reason": (
                "not applied because all training subgroups have equal image counts; "
                "subgroup balancing would be a constant multiplier"
            ),
        }

    if not estimator_accepts_sample_weight(model):
        model.fit(X_train, y_train)
        return {
            "sample_weight_used": 0,
            "sample_weight_reason": "estimator/version does not accept sample_weight",
        }

    try:
        if isinstance(model, Pipeline):
            model.fit(X_train, y_train, clf__sample_weight=raw_weights)
        else:
            model.fit(X_train, y_train, sample_weight=raw_weights)
        return {
            "sample_weight_used": 1,
            "sample_weight_reason": "subgroup-balanced sample weights applied",
        }
    except (TypeError, ValueError) as exc:
        if "sample_weight" not in str(exc):
            raise
        model.fit(X_train, y_train)
        return {
            "sample_weight_used": 0,
            "sample_weight_reason": f"sample_weight rejected; refit without weights: {exc}",
        }


# -----------------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------------


def default_feature_space() -> Dict[str, Dict[str, Any]]:
    return {
        "raw_stride8_gray": {"kind": "raw_stride", "color": "gray", "stride": 8},
        "raw_stride16_gray": {"kind": "raw_stride", "color": "gray", "stride": 16},
        "resize64_gray": {"kind": "resize_raw", "color": "gray", "size": 64},
        "resize128_gray": {"kind": "resize_raw", "color": "gray", "size": 128},
        "resize64_rgb": {"kind": "resize_raw", "color": "rgb", "size": 64},
        "hist_gray64": {"kind": "hist", "color": "gray", "bins": 64},
        "hist_rgb32": {"kind": "hist", "color": "rgb", "bins": 32},
        "hog128_gray": {
            "kind": "hog",
            "color": "gray",
            "size": 128,
            "orientations": 9,
            "pixels_per_cell": 16,
            "cells_per_block": 2,
        },
    }


def parse_features(features_arg: str) -> List[Tuple[str, Dict[str, Any]]]:
    space = default_feature_space()
    argument = features_arg.strip()
    if argument.lower() in {"all", "recommended", "default"}:
        return list(space.items())
    if argument.lower() == "fast":
        names = ["raw_stride16_gray", "resize64_gray", "hist_gray64", "hist_rgb32"]
        return [(name, space[name]) for name in names]
    names = [name.strip() for name in argument.split(",") if name.strip()]
    unknown = [name for name in names if name not in space]
    if unknown:
        raise ValueError(f"Unknown feature names: {unknown}; available={list(space)}")
    return [(name, space[name]) for name in names]


def pil_resample() -> int:
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def array_to_float01(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array.astype(np.float32) / 255.0
    if array.dtype == np.uint16:
        return array.astype(np.float32) / 65535.0
    result = array.astype(np.float32)
    finite = np.isfinite(result)
    if finite.any():
        minimum = float(result[finite].min())
        maximum = float(result[finite].max())
        if maximum > 1.0 or minimum < 0.0:
            denominator = maximum - minimum
            if denominator > 0:
                result = (result - minimum) / denominator
    result[~np.isfinite(result)] = 0.0
    return result


def load_image_array(path: str, color: str, resize: Optional[int] = None) -> np.ndarray:
    with Image.open(path) as image:
        if color == "gray":
            image = image.convert("L")
        elif color == "rgb":
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unknown color mode: {color}")
        if resize is not None:
            image = image.resize((resize, resize), pil_resample())
        array = np.asarray(image)
    return array_to_float01(array)


def extract_one_feature(path: str, spec: Dict[str, Any]) -> np.ndarray:
    kind = spec["kind"]
    color = spec.get("color", "gray")

    if kind == "raw_stride":
        stride = int(spec["stride"])
        array = load_image_array(path, color=color)
        array = array[::stride, ::stride] if array.ndim == 2 else array[::stride, ::stride, :]
        return array.reshape(-1).astype(np.float32)

    if kind == "resize_raw":
        size = int(spec["size"])
        return load_image_array(path, color=color, resize=size).reshape(-1).astype(np.float32)

    if kind == "hist":
        bins = int(spec["bins"])
        array = load_image_array(path, color=color)
        if array.ndim == 2:
            hist, _ = np.histogram(array.ravel(), bins=bins, range=(0.0, 1.0), density=True)
            return hist.astype(np.float32)
        channels = []
        for channel in range(array.shape[2]):
            hist, _ = np.histogram(
                array[:, :, channel].ravel(),
                bins=bins,
                range=(0.0, 1.0),
                density=True,
            )
            channels.append(hist.astype(np.float32))
        return np.concatenate(channels).astype(np.float32)

    if kind == "hog":
        if skimage_hog is None:
            raise RuntimeError("scikit-image is unavailable; install scikit-image")
        size = int(spec["size"])
        array = load_image_array(path, color="gray", resize=size)
        value = skimage_hog(
            array,
            orientations=int(spec["orientations"]),
            pixels_per_cell=(int(spec["pixels_per_cell"]), int(spec["pixels_per_cell"])),
            cells_per_block=(int(spec["cells_per_block"]), int(spec["cells_per_block"])),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return value.astype(np.float32)

    raise ValueError(f"Unknown feature kind: {kind}")


def feature_cache_key(paths: Sequence[str], feature_name: str, spec: Dict[str, Any]) -> str:
    digest = hashlib.sha1()
    digest.update(feature_name.encode("utf-8"))
    digest.update(json.dumps(spec, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        try:
            stat = os.stat(path)
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime)).encode("utf-8"))
        except OSError:
            pass
    return digest.hexdigest()[:20]


def extract_feature_matrix(
    paths: Sequence[str],
    feature_name: str,
    spec: Dict[str, Any],
    cache_dir: Path,
    use_cache: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{feature_name}_{feature_cache_key(paths, feature_name, spec)}.joblib"
    if use_cache and cache_path.exists():
        payload = joblib.load(cache_path)
        return payload["X"], payload["meta"]

    features: List[np.ndarray] = []
    lengths: Counter = Counter()
    failures: List[Tuple[str, str]] = []
    for index, path in enumerate(paths, start=1):
        try:
            feature = extract_one_feature(path, spec)
            features.append(feature)
            lengths[int(feature.size)] += 1
        except Exception as exc:
            failures.append((path, repr(exc)))
        if index % 1000 == 0:
            print(f"    extracted {index}/{len(paths)} images", flush=True)

    if failures:
        examples = "; ".join(f"{Path(path).name}: {error}" for path, error in failures[:5])
        raise RuntimeError(f"feature extraction failed for {len(failures)} files; {examples}")
    if len(lengths) != 1:
        raise RuntimeError(
            f"variable feature lengths for {feature_name}: {lengths.most_common(10)}"
        )

    X = np.vstack(features).astype(np.float32)
    metadata = {
        "feature": feature_name,
        "spec": spec,
        "n_images": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "dtype": str(X.dtype),
        "cache_path": str(cache_path),
    }
    if use_cache:
        joblib.dump({"X": X, "meta": metadata}, cache_path, compress=3)
    return X, metadata


# -----------------------------------------------------------------------------
# Grouping, folds, predictions, and metrics
# -----------------------------------------------------------------------------


def infer_subgroup_id(relative_path: str, sample_id: str, filename: str) -> str:
    rel = Path(str(relative_path))
    parent = rel.parent.as_posix() if str(rel.parent) != "." else str(sample_id)
    stem = Path(str(filename)).stem
    subgroup_stem = ANGLE_TOKEN_AT_END_RE.sub("", stem).rstrip("_- ")
    if not subgroup_stem:
        subgroup_stem = stem
    return f"{parent}/{subgroup_stem}"


def validate_manifest(df: pd.DataFrame, require_clean_profile: bool) -> pd.DataFrame:
    required = {
        "filepath",
        "relative_path",
        "filename",
        "sample_id_from_folder",
        "scenario1_label",
        "scenario2_label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")

    result = df.copy()
    if "qc_status" in result.columns:
        result = result[result["qc_status"].astype(str) == "accepted_tif"].copy()
    result = result.reset_index(drop=True)
    result["filepath"] = result["filepath"].astype(str)
    result["relative_path"] = result["relative_path"].astype(str)
    result["filename"] = result["filename"].astype(str)
    result["sample_id_from_folder"] = result["sample_id_from_folder"].astype(str).str.upper()

    if "subgroup_id" not in result.columns or result["subgroup_id"].astype(str).eq("").any():
        result["subgroup_id"] = [
            infer_subgroup_id(relative_path, sample_id, filename)
            for relative_path, sample_id, filename in zip(
                result["relative_path"],
                result["sample_id_from_folder"],
                result["filename"],
            )
        ]
    result["subgroup_id"] = result["subgroup_id"].astype(str)

    missing_paths = [path for path in result["filepath"] if not Path(path).exists()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest paths are missing; example={missing_paths[0]}"
        )

    duplicates = result["filepath"].duplicated().sum()
    if duplicates:
        raise ValueError(f"manifest contains {duplicates} duplicate filepath rows")

    for scenario, info in SCENARIOS.items():
        labels = set(result[info["label_col"]].dropna().astype(str))
        unknown = sorted(labels - set(info["classes"]))
        if unknown:
            raise ValueError(f"{scenario} contains unknown labels: {unknown}")

    group_label_counts = (
        result.groupby("subgroup_id")[["scenario1_label", "scenario2_label"]]
        .nunique(dropna=False)
    )
    mixed = group_label_counts[(group_label_counts > 1).any(axis=1)]
    if not mixed.empty:
        raise ValueError(f"mixed-label subgroups found: {mixed.head(10).to_dict('index')}")

    if require_clean_profile:
        failures = []
        group_sizes = result.groupby("subgroup_id").size()
        if len(result) != EXPECTED_CLEAN_PROFILE["n_images"]:
            failures.append(f"n_images={len(result)}")
        if result["subgroup_id"].nunique() != EXPECTED_CLEAN_PROFILE["n_subgroups"]:
            failures.append(f"n_subgroups={result['subgroup_id'].nunique()}")
        if result["sample_id_from_folder"].nunique() != EXPECTED_CLEAN_PROFILE["n_sample_ids"]:
            failures.append(f"n_sample_ids={result['sample_id_from_folder'].nunique()}")
        bad_sizes = group_sizes[group_sizes != EXPECTED_CLEAN_PROFILE["group_size"]]
        if not bad_sizes.empty:
            failures.append(
                f"bad_group_sizes={bad_sizes.head(10).to_dict()} total={len(bad_sizes)}"
            )
        if failures:
            raise ValueError(
                "clean-profile validation failed: " + "; ".join(failures)
            )
    return result


def make_group_table(sdf: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for group_id, group_df in sdf.groupby("subgroup_id", sort=True):
        labels = group_df[label_col].astype(str).unique().tolist()
        samples = group_df["sample_id_from_folder"].astype(str).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"subgroup {group_id} has mixed labels: {labels}")
        if len(samples) != 1:
            raise ValueError(f"subgroup {group_id} has mixed sample IDs: {samples}")
        rows.append(
            {
                "subgroup_id": str(group_id),
                "sample_id": samples[0],
                "label": labels[0],
                "n_images": int(len(group_df)),
            }
        )
    return pd.DataFrame(rows)


def build_fold_assignment(
    group_table: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    counts = Counter(group_table["label"].astype(str))
    if min(counts.values()) < n_splits:
        raise ValueError(
            f"minimum subgroup class count={min(counts.values())} < folds={n_splits}; counts={dict(counts)}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = group_table.copy()
    result["fold"] = 0
    group_indices = np.arange(len(result))
    labels = result["label"].astype(str).to_numpy()
    for fold, (_, test_indices) in enumerate(splitter.split(group_indices, labels), start=1):
        result.loc[test_indices, "fold"] = fold
    if (result["fold"] == 0).any():
        raise RuntimeError("some subgroups were not assigned to a fold")
    return result.sort_values(["fold", "label", "subgroup_id"]).reset_index(drop=True)


def indices_for_groups(groups: np.ndarray, selected: Sequence[str]) -> np.ndarray:
    selected_set = set(map(str, selected))
    return np.flatnonzero(np.isin(groups.astype(str), list(selected_set)))


def image_score_matrix(
    model: Any,
    X: np.ndarray,
    class_order: Sequence[str],
) -> Tuple[np.ndarray, str]:
    model_classes = get_model_classes(model)
    class_to_column = {label: index for index, label in enumerate(class_order)}

    if hasattr(model, "predict_proba"):
        try:
            raw = np.asarray(model.predict_proba(X), dtype=np.float64)
            aligned = np.full((len(X), len(class_order)), -np.inf, dtype=np.float64)
            for source_column, label in enumerate(model_classes):
                if label in class_to_column:
                    aligned[:, class_to_column[label]] = raw[:, source_column]
            return aligned, "probability"
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            raw = np.asarray(model.decision_function(X), dtype=np.float64)
            if raw.ndim == 1:
                if len(model_classes) != 2:
                    raise ValueError("binary decision scores but model classes are unavailable")
                raw = np.column_stack([-raw, raw])
            aligned = np.full((len(X), len(class_order)), -np.inf, dtype=np.float64)
            for source_column, label in enumerate(model_classes):
                if label in class_to_column:
                    aligned[:, class_to_column[label]] = raw[:, source_column]
            return aligned, "decision_score"
        except Exception:
            pass

    predictions = np.asarray(model.predict(X)).astype(str)
    aligned = np.zeros((len(X), len(class_order)), dtype=np.float64)
    for row, label in enumerate(predictions):
        if label in class_to_column:
            aligned[row, class_to_column[label]] = 1.0
    return aligned, "hard_vote"


def aggregate_subgroups(
    y_true: np.ndarray,
    groups: np.ndarray,
    image_scores: np.ndarray,
    class_order: Sequence[str],
    fold: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    class_array = np.asarray(class_order, dtype=object)
    for group_id in sorted(set(groups.astype(str))):
        mask = groups.astype(str) == group_id
        true_labels = np.unique(y_true[mask].astype(str))
        if len(true_labels) != 1:
            raise ValueError(f"OOF subgroup {group_id} has mixed labels: {true_labels}")
        average_scores = np.mean(image_scores[mask], axis=0)
        predicted_index = int(np.argmax(average_scores))
        predicted_label = str(class_array[predicted_index])
        row: Dict[str, Any] = {
            "fold": int(fold),
            "subgroup_id": group_id,
            "y_true": str(true_labels[0]),
            "y_pred": predicted_label,
            "correct": int(str(true_labels[0]) == predicted_label),
            "n_images": int(mask.sum()),
            "winning_score": float(average_scores[predicted_index]),
        }
        for class_index, class_label in enumerate(class_order):
            row[f"score_{safe_name(class_label)}"] = float(average_scores[class_index])
        rows.append(row)
    return pd.DataFrame(rows)


def metric_dictionary(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_order: Sequence[str],
    prefix: str,
) -> Dict[str, float]:
    true_array = np.asarray(y_true).astype(str)
    pred_array = np.asarray(y_pred).astype(str)
    result: Dict[str, float] = {
        f"{prefix}_acc": float(accuracy_score(true_array, pred_array)),
        f"{prefix}_macro_f1": float(
            f1_score(true_array, pred_array, labels=list(class_order), average="macro", zero_division=0)
        ),
        f"{prefix}_weighted_f1": float(
            f1_score(true_array, pred_array, labels=list(class_order), average="weighted", zero_division=0)
        ),
    }
    f1_values = f1_score(
        true_array, pred_array, labels=list(class_order), average=None, zero_division=0
    )
    precision_values = precision_score(
        true_array, pred_array, labels=list(class_order), average=None, zero_division=0
    )
    recall_values = recall_score(
        true_array, pred_array, labels=list(class_order), average=None, zero_division=0
    )
    for label, f1_value, precision_value, recall_value in zip(
        class_order, f1_values, precision_values, recall_values
    ):
        key = safe_name(label)
        mask = true_array == label
        class_accuracy = float(np.mean(pred_array[mask] == true_array[mask])) if mask.any() else float("nan")
        result[f"{prefix}_{key}_acc"] = class_accuracy
        result[f"{prefix}_{key}_f1"] = float(f1_value)
        result[f"{prefix}_{key}_precision"] = float(precision_value)
        result[f"{prefix}_{key}_recall"] = float(recall_value)
    return result


def per_class_rows(
    scenario: str,
    feature: str,
    model: str,
    level: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_order: Sequence[str],
) -> List[Dict[str, Any]]:
    true_array = np.asarray(y_true).astype(str)
    pred_array = np.asarray(y_pred).astype(str)
    f1_values = f1_score(true_array, pred_array, labels=list(class_order), average=None, zero_division=0)
    precision_values = precision_score(true_array, pred_array, labels=list(class_order), average=None, zero_division=0)
    recall_values = recall_score(true_array, pred_array, labels=list(class_order), average=None, zero_division=0)
    rows = []
    for label, f1_value, precision_value, recall_value in zip(
        class_order, f1_values, precision_values, recall_values
    ):
        mask = true_array == label
        rows.append(
            {
                "scenario": scenario,
                "feature": feature,
                "model": model,
                "level": level,
                "class": label,
                "support": int(mask.sum()),
                "accuracy_recall": float(recall_value),
                "precision": float(precision_value),
                "recall": float(recall_value),
                "f1": float(f1_value),
            }
        )
    return rows


def confusion_rows(
    scenario: str,
    feature: str,
    model: str,
    level: str,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_order: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matrix = confusion_matrix(y_true, y_pred, labels=list(class_order))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )
    counts_rows: List[Dict[str, Any]] = []
    normalized_rows: List[Dict[str, Any]] = []
    for true_index, true_label in enumerate(class_order):
        for pred_index, predicted_label in enumerate(class_order):
            common = {
                "scenario": scenario,
                "feature": feature,
                "model": model,
                "level": level,
                "true_label": true_label,
                "predicted_label": predicted_label,
            }
            counts_rows.append({**common, "count": int(matrix[true_index, pred_index])})
            normalized_rows.append(
                {**common, "row_normalized_fraction": float(normalized[true_index, pred_index])}
            )
    return counts_rows, normalized_rows


# -----------------------------------------------------------------------------
# One configuration
# -----------------------------------------------------------------------------


def run_configuration(
    X: np.ndarray,
    sdf: pd.DataFrame,
    scenario: str,
    label_col: str,
    class_order: Sequence[str],
    fold_assignment: pd.DataFrame,
    feature_name: str,
    feature_spec: Dict[str, Any],
    model_name: str,
    out_dir: Path,
    seed: int,
    n_estimators: int,
    max_iter: int,
    scale_dim_limit: int,
    save_oof: bool,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    y = sdf[label_col].astype(str).to_numpy()
    groups = sdf["subgroup_id"].astype(str).to_numpy()
    skip_reason = should_skip_model(model_name, n_train=int(len(y) * 0.8), dim=X.shape[1])
    base = {
        "scenario": scenario,
        "feature": feature_name,
        "model": model_name,
        "n_images": int(len(y)),
        "n_subgroups": int(len(set(groups))),
        "n_features": int(X.shape[1]),
        "class_counts": json.dumps(dict(Counter(y)), sort_keys=True),
        "feature_spec": json.dumps(feature_spec, sort_keys=True),
    }
    if skip_reason:
        return (
            {**base, "cv_status": "skipped", "cv_reason": skip_reason},
            [],
            [],
            [],
            [],
        )

    config_dir = out_dir / scenario / "configs" / f"{safe_name(feature_name)}__{safe_name(model_name)}"
    config_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, Any]] = []
    image_oof_parts: List[pd.DataFrame] = []
    group_oof_parts: List[pd.DataFrame] = []
    weight_records: List[Dict[str, Any]] = []

    for fold in sorted(fold_assignment["fold"].unique()):
        test_groups = fold_assignment.loc[
            fold_assignment["fold"] == fold, "subgroup_id"
        ].astype(str).tolist()
        train_groups = fold_assignment.loc[
            fold_assignment["fold"] != fold, "subgroup_id"
        ].astype(str).tolist()
        train_indices = indices_for_groups(groups, train_groups)
        test_indices = indices_for_groups(groups, test_groups)

        overlap = set(groups[train_indices]).intersection(set(groups[test_indices]))
        if overlap:
            raise RuntimeError(f"group leakage in fold {fold}: {list(overlap)[:5]}")
        if len(train_indices) + len(test_indices) != len(y):
            raise RuntimeError(f"fold {fold} does not cover every image")

        model, scaler_used = build_model_pipeline(
            model_name=model_name,
            seed=seed + int(fold),
            n_estimators=n_estimators,
            max_iter=max_iter,
            dim=X.shape[1],
            scale_dim_limit=scale_dim_limit,
        )
        weight_status = fit_model(
            model,
            X[train_indices],
            y[train_indices],
            groups[train_indices],
        )
        weight_records.append({"fold": int(fold), **weight_status})

        image_predictions = np.asarray(model.predict(X[test_indices])).astype(str)
        image_scores, score_type = image_score_matrix(model, X[test_indices], class_order)

        image_oof = sdf.iloc[test_indices][
            [
                "filepath",
                "relative_path",
                "filename",
                "sample_id_from_folder",
                "subgroup_id",
            ]
        ].copy()
        image_oof.insert(0, "fold", int(fold))
        image_oof["y_true"] = y[test_indices]
        image_oof["y_pred"] = image_predictions
        image_oof["correct"] = (y[test_indices] == image_predictions).astype(int)
        image_oof["score_type"] = score_type
        for class_index, class_label in enumerate(class_order):
            image_oof[f"score_{safe_name(class_label)}"] = image_scores[:, class_index]
        image_oof_parts.append(image_oof)

        group_oof = aggregate_subgroups(
            y_true=y[test_indices],
            groups=groups[test_indices],
            image_scores=image_scores,
            class_order=class_order,
            fold=int(fold),
        )
        group_to_sample = (
            sdf.iloc[test_indices]
            .groupby("subgroup_id")["sample_id_from_folder"]
            .first()
            .to_dict()
        )
        group_oof.insert(
            2,
            "sample_id",
            group_oof["subgroup_id"].map(group_to_sample),
        )
        group_oof["score_type"] = score_type
        group_oof_parts.append(group_oof)

        fold_row = {
            **base,
            "fold": int(fold),
            "n_train_images": int(len(train_indices)),
            "n_test_images": int(len(test_indices)),
            "n_train_groups": int(len(train_groups)),
            "n_test_groups": int(len(test_groups)),
            "scaler_used": int(scaler_used),
            **weight_status,
            "subgroup_aggregation": score_type,
        }
        fold_row.update(
            metric_dictionary(
                y[test_indices], image_predictions, class_order, "fold_image"
            )
        )
        fold_row.update(
            metric_dictionary(
                group_oof["y_true"], group_oof["y_pred"], class_order, "fold_group"
            )
        )
        fold_rows.append(fold_row)

    image_oof_all = pd.concat(image_oof_parts, ignore_index=True).sort_values(
        ["fold", "subgroup_id", "filename"]
    )
    group_oof_all = pd.concat(group_oof_parts, ignore_index=True).sort_values(
        ["fold", "subgroup_id"]
    )

    if len(image_oof_all) != len(sdf):
        raise RuntimeError(
            f"image OOF coverage={len(image_oof_all)} expected={len(sdf)}"
        )
    if image_oof_all["filepath"].duplicated().any():
        raise RuntimeError("some images have multiple OOF predictions")
    expected_groups = sdf["subgroup_id"].nunique()
    if len(group_oof_all) != expected_groups:
        raise RuntimeError(
            f"group OOF coverage={len(group_oof_all)} expected={expected_groups}"
        )
    if group_oof_all["subgroup_id"].duplicated().any():
        raise RuntimeError("some subgroups have multiple OOF predictions")

    summary: Dict[str, Any] = {
        **base,
        "cv_status": "ok",
        "cv_reason": "",
        "folds_completed": int(len(fold_rows)),
        "scaler_used_any_fold": int(any(row["scaler_used"] for row in fold_rows)),
        "sample_weight_used_folds": int(sum(row["sample_weight_used"] for row in fold_rows)),
        "sample_weight_notes": " | ".join(
            sorted(set(row["sample_weight_reason"] for row in fold_rows))
        ),
        "subgroup_aggregation_methods": "|".join(
            sorted(set(row["subgroup_aggregation"] for row in fold_rows))
        ),
    }

    metric_keys = [
        "fold_image_acc",
        "fold_image_macro_f1",
        "fold_image_weighted_f1",
        "fold_group_acc",
        "fold_group_macro_f1",
        "fold_group_weighted_f1",
    ]
    for key in metric_keys:
        mean, std = mean_std(row[key] for row in fold_rows)
        output_key = "cv_" + key.removeprefix("fold_")
        summary[f"{output_key}_mean"] = mean
        summary[f"{output_key}_std"] = std

    summary.update(
        metric_dictionary(
            image_oof_all["y_true"],
            image_oof_all["y_pred"],
            class_order,
            "oof_image",
        )
    )
    summary.update(
        metric_dictionary(
            group_oof_all["y_true"],
            group_oof_all["y_pred"],
            class_order,
            "oof_group",
        )
    )

    per_class = []
    per_class.extend(
        per_class_rows(
            scenario,
            feature_name,
            model_name,
            "image",
            image_oof_all["y_true"],
            image_oof_all["y_pred"],
            class_order,
        )
    )
    per_class.extend(
        per_class_rows(
            scenario,
            feature_name,
            model_name,
            "subgroup",
            group_oof_all["y_true"],
            group_oof_all["y_pred"],
            class_order,
        )
    )

    confusion_counts: List[Dict[str, Any]] = []
    confusion_normalized: List[Dict[str, Any]] = []
    for level, true_values, predicted_values in [
        ("image", image_oof_all["y_true"], image_oof_all["y_pred"]),
        ("subgroup", group_oof_all["y_true"], group_oof_all["y_pred"]),
    ]:
        count_rows, normalized_rows = confusion_rows(
            scenario,
            feature_name,
            model_name,
            level,
            true_values,
            predicted_values,
            class_order,
        )
        confusion_counts.extend(count_rows)
        confusion_normalized.extend(normalized_rows)

    pd.DataFrame(fold_rows).to_csv(config_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(config_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_counts).to_csv(
        config_dir / "confusion_matrix_counts.csv", index=False
    )
    pd.DataFrame(confusion_normalized).to_csv(
        config_dir / "confusion_matrix_row_normalized.csv", index=False
    )
    if save_oof:
        image_oof_all.to_csv(
            config_dir / "oof_image_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
        group_oof_all.to_csv(
            config_dir / "oof_subgroup_predictions.csv.gz",
            index=False,
            compression="gzip",
        )
    json_dump(
        config_dir / "metadata.json",
        {
            "scenario": scenario,
            "feature": feature_name,
            "feature_spec": feature_spec,
            "model": model_name,
            "class_order": list(class_order),
            "summary": summary,
            "weight_records": weight_records,
            "note": "Five-fold group-aware OOF evaluation only; no holdout and no final model fit.",
        },
    )

    return summary, fold_rows, per_class, confusion_counts, confusion_normalized


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenarios", default="scenario1,scenario2")
    parser.add_argument("--features", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-iter", type=int, default=8000)
    parser.add_argument("--scale-dim-limit", type=int, default=400_000)
    parser.add_argument("--best-metric", default="cv_group_macro_f1_mean")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-oof", action="store_true")
    parser.add_argument("--require-clean-profile", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "feature_cache"

    scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    invalid_scenarios = [name for name in scenario_names if name not in SCENARIOS]
    if invalid_scenarios:
        raise ValueError(f"unknown scenarios: {invalid_scenarios}")

    feature_items = parse_features(args.features)
    model_names = parse_models(args.models)
    if not feature_items:
        raise ValueError("no features selected")
    if not model_names:
        raise ValueError("no models selected")

    manifest = validate_manifest(
        pd.read_csv(manifest_path),
        require_clean_profile=args.require_clean_profile,
    )
    paths = manifest["filepath"].astype(str).tolist()

    run_settings = {
        "stage": "group_aware_5fold_internal_validation",
        "manifest": str(manifest_path),
        "out": str(out_dir),
        "scenarios": scenario_names,
        "features": [name for name, _ in feature_items],
        "models": model_names,
        "folds": args.folds,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "max_iter": args.max_iter,
        "scale_dim_limit": args.scale_dim_limit,
        "best_metric": args.best_metric,
        "save_oof": not args.no_oof,
        "require_clean_profile": args.require_clean_profile,
        "split_unit": "subgroup",
        "holdout": "not performed",
        "final_model": "not fitted",
    }
    json_dump(out_dir / "run_settings.json", run_settings)
    json_dump(out_dir / "environment.json", environment_information())
    manifest.to_csv(out_dir / "manifest_used.csv", index=False)

    all_results: List[Dict[str, Any]] = []
    all_folds: List[Dict[str, Any]] = []
    all_per_class: List[Dict[str, Any]] = []
    all_confusion_counts: List[Dict[str, Any]] = []
    all_confusion_normalized: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # Feature extraction is independent of scenario and is done once.
    for feature_name, feature_spec in feature_items:
        print(f"\n===== FEATURE {feature_name} =====", flush=True)
        try:
            X, feature_metadata = extract_feature_matrix(
                paths=paths,
                feature_name=feature_name,
                spec=feature_spec,
                cache_dir=cache_dir,
                use_cache=not args.no_cache,
            )
            print(f"Feature matrix: {X.shape}", flush=True)
            json_dump(out_dir / f"feature_metadata_{safe_name(feature_name)}.json", feature_metadata)
        except Exception as exc:
            reason = repr(exc)
            print(f"FEATURE ERROR {feature_name}: {reason}", flush=True)
            for scenario in scenario_names:
                for model_name in model_names:
                    all_results.append(
                        {
                            "scenario": scenario,
                            "feature": feature_name,
                            "model": model_name,
                            "cv_status": "feature_error",
                            "cv_reason": reason,
                        }
                    )
            errors.append(
                {
                    "scenario": "all",
                    "feature": feature_name,
                    "model": "all",
                    "error": reason,
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        for scenario in scenario_names:
            info = SCENARIOS[scenario]
            label_col = info["label_col"]
            class_order = info["classes"]
            scenario_mask = manifest[label_col].notna() & manifest[label_col].astype(str).isin(class_order)
            sdf = manifest.loc[scenario_mask].reset_index(drop=True)
            scenario_indices = manifest.index[scenario_mask].to_numpy()
            scenario_X = X[scenario_indices]

            group_table = make_group_table(sdf, label_col)
            fold_assignment = build_fold_assignment(group_table, args.folds, args.seed)
            scenario_dir = out_dir / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            fold_assignment.to_csv(scenario_dir / "fold_assignments.csv", index=False)
            sdf.to_csv(scenario_dir / "manifest_used.csv", index=False)

            print(
                f"\n[{scenario}] images={len(sdf)} subgroups={len(group_table)} "
                f"class_images={dict(Counter(sdf[label_col].astype(str)))} "
                f"class_subgroups={dict(Counter(group_table['label'].astype(str)))}",
                flush=True,
            )

            for model_name in model_names:
                print(f"  {feature_name} + {model_name}", flush=True)
                try:
                    summary, fold_rows, per_class, counts, normalized = run_configuration(
                        X=scenario_X,
                        sdf=sdf,
                        scenario=scenario,
                        label_col=label_col,
                        class_order=class_order,
                        fold_assignment=fold_assignment,
                        feature_name=feature_name,
                        feature_spec=feature_spec,
                        model_name=model_name,
                        out_dir=out_dir,
                        seed=args.seed,
                        n_estimators=args.n_estimators,
                        max_iter=args.max_iter,
                        scale_dim_limit=args.scale_dim_limit,
                        save_oof=not args.no_oof,
                    )
                    all_results.append(summary)
                    all_folds.extend(fold_rows)
                    all_per_class.extend(per_class)
                    all_confusion_counts.extend(counts)
                    all_confusion_normalized.extend(normalized)
                    if summary.get("cv_status") == "ok":
                        print(
                            "    DONE "
                            f"group Macro-F1={summary.get('cv_group_macro_f1_mean', float('nan')):.4f} "
                            f"image Macro-F1={summary.get('cv_image_macro_f1_mean', float('nan')):.4f}",
                            flush=True,
                        )
                    else:
                        print(f"    SKIP {summary.get('cv_reason')}", flush=True)
                except Exception as exc:
                    error_row = {
                        "scenario": scenario,
                        "feature": feature_name,
                        "model": model_name,
                        "cv_status": "error",
                        "cv_reason": repr(exc),
                    }
                    all_results.append(error_row)
                    errors.append(
                        {
                            **error_row,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"    ERROR {exc}", flush=True)

                pd.DataFrame(all_results).to_csv(
                    out_dir / "results_summary_partial.csv", index=False
                )
                if all_folds:
                    pd.DataFrame(all_folds).to_csv(
                        out_dir / "fold_results_partial.csv", index=False
                    )
                if errors:
                    pd.DataFrame(errors).to_csv(
                        out_dir / "errors_partial.csv", index=False
                    )

    results_df = pd.DataFrame(all_results)
    folds_df = pd.DataFrame(all_folds)
    per_class_df = pd.DataFrame(all_per_class)
    confusion_counts_df = pd.DataFrame(all_confusion_counts)
    confusion_normalized_df = pd.DataFrame(all_confusion_normalized)
    errors_df = pd.DataFrame(errors)

    results_df.to_csv(out_dir / "results_summary.csv", index=False)
    folds_df.to_csv(out_dir / "fold_results.csv", index=False)
    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    confusion_counts_df.to_csv(out_dir / "confusion_matrix_counts.csv", index=False)
    confusion_normalized_df.to_csv(
        out_dir / "confusion_matrix_row_normalized.csv", index=False
    )
    if not errors_df.empty:
        errors_df.to_csv(out_dir / "errors.csv", index=False)

    skipped = results_df[results_df.get("cv_status", pd.Series(dtype=str)) != "ok"].copy()
    skipped.to_csv(out_dir / "skipped_or_failed_configurations.csv", index=False)

    if args.best_metric not in results_df.columns:
        valid = pd.DataFrame()
    else:
        valid = results_df[
            (results_df["cv_status"] == "ok")
            & pd.to_numeric(results_df[args.best_metric], errors="coerce").notna()
        ].copy()
    if not valid.empty:
        valid[args.best_metric] = pd.to_numeric(valid[args.best_metric], errors="coerce")
        ranked = valid.sort_values(
            ["scenario", args.best_metric, "feature", "model"],
            ascending=[True, False, True, True],
        )
        ranked["rank_within_scenario"] = ranked.groupby("scenario").cumcount() + 1
        ranked.to_csv(out_dir / "ranked_results.csv", index=False)
        ranked.groupby("scenario", group_keys=False).head(1).to_csv(
            out_dir / "best_per_scenario.csv", index=False
        )
        ranked.groupby("scenario", group_keys=False).head(3).to_csv(
            out_dir / "top3_per_scenario.csv", index=False
        )

    print("\nCOMPLETE", flush=True)
    print(f"Results: {out_dir / 'results_summary.csv'}", flush=True)
    print(f"Folds:   {out_dir / 'fold_results.csv'}", flush=True)
    print(f"OOF:     {out_dir}/<scenario>/configs/<feature>__<model>/", flush=True)
    print("No 70/30 holdout was performed. No final model was fitted.", flush=True)


if __name__ == "__main__":
    main()
