#!/usr/bin/env python3
"""
BF/ED paired RAW-PIXEL image classification with 3-fold/repeated-split evaluation.

Purpose
-------
Input during training:
    BF image + ED image pair from cancer_combine_new only

Input modes evaluated:
    BF       : BF raw pixels only
    ED       : ED raw pixels only
    BFplusED : concatenated [BF raw pixels | ED raw pixels]

Output classes by default:
    HPNE, PANC, MIA

Main improvements over the old single-split script
--------------------------------------------------
1. Uses 3-fold cross-validation by default, not one lucky train/test split.
   Also supports 3 repeated stratified shuffle splits with user-selected --test-size,
   e.g., --cv 3 --test-size 0.3, 0.5, or 0.7.
2. Reports Macro-F1 and overall accuracy.
3. Reports each-class accuracy and each-class F1 for HPNE, PANC, and MIA.
4. Produces easy paper tables across BF, ED, and BFplusED.
5. Uses train-fold-only StandardScaler inside sklearn Pipeline when scaling is enabled,
   so there is no scaler leakage across folds.
6. Optional group-aware CV keeps duplicate/paired sample keys in the same fold.

Important note about class accuracy
-----------------------------------
Per-class accuracy here means:
    class_correct / total_true_samples_of_that_class
This is the same quantity as class recall, but it is labeled ClassAcc because it is
what many biomedical-image papers call class-wise accuracy.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from tqdm import tqdm

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.1
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None  # type: ignore

HAS_MPL = False
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None
    HAS_MPL = False

# =============================================================================
# CLASS ORDER: 3-class default for your current task
# =============================================================================
CLASS_ORDER = ["HPNE", "PANC", "MIA"]

# =============================================================================
# DATASET PATHS: copied from your existing script
# =============================================================================
# Publication-safe data location: set CANCER_DATA_ROOT to the directory
# containing cancer_combine_new/. The original paper used this same
# relative directory layout; pairing, preprocessing, CV and scoring are unchanged.
DATA_ROOT = Path(os.environ.get("CANCER_DATA_ROOT", "./private_data"))

DATA_PAIRED: Dict[str, Dict[str, List[Path]]] = {
    # New paired dataset ONLY.
    # Pairing rule:
    #   <CLASS>_BrightField01.tiff  <->  <CLASS>_ED0001.tiff
    #   <CLASS>_BrightField02.tiff  <->  <CLASS>_ED0002.tiff
    # sample_key_from_name() maps both names to <CLASS>_0001, <CLASS>_0002, etc.
    "HPNE": {
        "BF": [
            DATA_ROOT / "cancer_combine_new/New_HPNE_Bf/New_HPNE_Bf",
        ],
        "ED": [
            DATA_ROOT / "cancer_combine_new/New_HPNE_ED_Result/New_HPNE_ED_Result",
        ],
    },
    "PANC": {
        "BF": [
            DATA_ROOT / "cancer_combine_new/New_PANC_Bf/New_PANC_Bf",
        ],
        "ED": [
            DATA_ROOT / "cancer_combine_new/New_PANC_ED_Result/New_PANC_ED_Images",
        ],
    },
    "MIA": {
        "BF": [
            DATA_ROOT / "cancer_combine_new/New_MIA_Bf/New_MIA_Bf",
        ],
        "ED": [
            DATA_ROOT / "cancer_combine_new/New_MIA_ED_Result/New_MIA_ED_Result",
        ],
    },
}

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# =============================================================================
# Image loading backends
# =============================================================================
try:
    import tifffile  # type: ignore
except Exception:  # pragma: no cover
    tifffile = None  # type: ignore

try:
    import imageio.v3 as iio  # type: ignore
except Exception:  # pragma: no cover
    iio = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image_any(path: Path) -> np.ndarray:
    if tifffile is not None:
        return tifffile.imread(str(path))
    if iio is not None:
        return iio.imread(str(path))
    if Image is not None:
        return np.array(Image.open(path))
    raise RuntimeError("No image reader installed. Install tifffile, imageio, or pillow.")


# =============================================================================
# Pairing by sample key
# =============================================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def sample_key_from_name(fname: str) -> str:
    """
    Supports both naming systems from the old script.

    New:
        HPNE_BrightField01.tiff -> HPNE_0001
        HPNE_ED0001.tiff        -> HPNE_0001
        PANC_BrightField02.tiff -> PANC_0002
        PANC_ED0002.tiff        -> PANC_0002

    Old fallback:
        HPNE_10_20x_ang0.tif -> HPNE_10
    """
    stem = Path(fname).stem

    m = re.match(r"^(HPNE|MIA|PANC)_BrightField(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        cls = m.group(1).upper()
        idx = int(m.group(2))
        return f"{cls}_{idx:04d}"

    m = re.match(r"^(HPNE|MIA|PANC)_ED0*(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        cls = m.group(1).upper()
        idx = int(m.group(2))
        return f"{cls}_{idx:04d}"

    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    return f"{parts[0]}_{parts[1]}"


def group_key_from_pair_key(pair_key: str) -> str:
    """Remove duplicate suffix like __01 so repeated images from same sample stay grouped."""
    return re.sub(r"__\d+$", "", pair_key)


def index_dir_by_key(folder: Path, recursive: bool) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    if not folder.exists():
        print(f"[WARN] Folder not found: {folder}")
        return out

    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    for path in sorted(iterator):
        if not is_image_file(path):
            continue
        key = sample_key_from_name(path.name)
        out.setdefault(key, []).append(path)

    for key in out:
        out[key] = sorted(out[key])
    return out


def merge_folder_maps(folders: List[Path], recursive: bool) -> Dict[str, List[Path]]:
    merged: Dict[str, List[Path]] = {}
    for folder in folders:
        one = index_dir_by_key(folder, recursive=recursive)
        for key, paths in one.items():
            merged.setdefault(key, []).extend(paths)

    for key in merged:
        merged[key] = sorted(merged[key])
    return merged


# =============================================================================
# Raw-pixel extraction
# =============================================================================
def _dtype_scale(arr: np.ndarray) -> float:
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return float(info.max) if info.max > 0 else 1.0
    return 1.0


def bf_raw_pixels(bf_path: Path, stride: int) -> np.ndarray:
    arr = read_image_any(bf_path)

    # Force HWC RGB with 3 channels.
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))[..., :3]
    else:
        raise ValueError(f"Unsupported BF shape {arr.shape} for {bf_path}")

    if stride > 1:
        arr = arr[::stride, ::stride, :]

    x = arr.astype(np.float32, copy=False)
    scale = _dtype_scale(arr)
    if scale != 1.0:
        x = x / scale

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)


def ed_raw_pixels(ed_path: Path, stride: int, ed_norm: str) -> np.ndarray:
    arr = read_image_any(ed_path)

    # Force 2D ED image.
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., 0]
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Unsupported ED shape {arr.shape} for {ed_path}")

    if stride > 1:
        arr = arr[::stride, ::stride]

    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if ed_norm == "none":
        pass
    elif ed_norm == "dtype":
        scale = _dtype_scale(arr)
        if scale != 1.0:
            x = x / scale
    elif ed_norm == "minmax":
        finite = np.isfinite(x)
        if np.any(finite):
            lo = float(np.min(x[finite]))
            hi = float(np.max(x[finite]))
            x = (x - lo) / (hi - lo) if hi > lo else x * 0.0
        else:
            x = x * 0.0
    else:
        raise ValueError(f"Unknown ed_norm={ed_norm}")

    return x.reshape(-1)


def build_paired_dataset(
    class_names: Sequence[str],
    recursive: bool,
    max_per_class: int,
    stride: int,
    ed_norm: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    label_map = {name: idx for idx, name in enumerate(class_names)}
    items: List[Tuple[str, Path, Path, int]] = []

    print("[INFO] Pairing BF/ED images by sample key...")
    for cls in class_names:
        if cls not in DATA_PAIRED:
            raise RuntimeError(f"Class {cls} is not defined in DATA_PAIRED.")

        bf_map = merge_folder_maps(DATA_PAIRED[cls]["BF"], recursive=recursive)
        ed_map = merge_folder_maps(DATA_PAIRED[cls]["ED"], recursive=recursive)
        common_keys = sorted(set(bf_map.keys()) & set(ed_map.keys()))

        cls_items: List[Tuple[str, Path, Path, int]] = []
        paired_total = 0
        for key in common_keys:
            n = min(len(bf_map[key]), len(ed_map[key]))
            for i in range(n):
                pair_key = f"{key}__{i + 1:02d}"
                cls_items.append((pair_key, bf_map[key][i], ed_map[key][i], label_map[cls]))
                paired_total += 1

        if max_per_class > 0:
            cls_items = cls_items[:max_per_class]

        print(
            f"[INFO] {cls}: BF_keys={len(bf_map)} ED_keys={len(ed_map)} "
            f"paired={paired_total} used={len(cls_items)}"
        )
        items.extend(cls_items)

    if not items:
        raise RuntimeError("No paired samples found. Check folder paths, naming, and --recursive.")

    x_bf_list: List[np.ndarray] = []
    x_ed_list: List[np.ndarray] = []
    y_list: List[int] = []
    key_list: List[str] = []

    expected_bf_len: Optional[int] = None
    expected_ed_len: Optional[int] = None

    print(f"[INFO] Extracting raw pixels from {len(items)} pairs with stride={stride}...")
    for pair_key, bf_path, ed_path, yi in tqdm(items, desc="raw_pixels"):
        try:
            bf_vec = bf_raw_pixels(bf_path, stride=stride)
            ed_vec = ed_raw_pixels(ed_path, stride=stride, ed_norm=ed_norm)
        except Exception as exc:
            print(f"[WARN] Skipping {pair_key}: {exc}")
            continue

        if expected_bf_len is None:
            expected_bf_len = int(bf_vec.shape[0])
            expected_ed_len = int(ed_vec.shape[0])
            print(f"[INFO] Expected vector lengths: BF={expected_bf_len}, ED={expected_ed_len}")

        if bf_vec.shape[0] != expected_bf_len or ed_vec.shape[0] != expected_ed_len:
            print(
                f"[WARN] Skipping {pair_key}: size mismatch BF={bf_vec.shape[0]} ED={ed_vec.shape[0]} "
                f"expected BF={expected_bf_len} ED={expected_ed_len}"
            )
            continue

        x_bf_list.append(bf_vec.astype(np.float32, copy=False))
        x_ed_list.append(ed_vec.astype(np.float32, copy=False))
        y_list.append(int(yi))
        key_list.append(pair_key)

    if not x_bf_list:
        raise RuntimeError("All samples failed raw-pixel extraction.")

    x_bf = np.stack(x_bf_list, axis=0).astype(np.float32, copy=False)
    x_ed = np.stack(x_ed_list, axis=0).astype(np.float32, copy=False)
    x_bfed = np.concatenate([x_bf, x_ed], axis=1).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    keys = np.asarray(key_list, dtype=object)

    print("[INFO] Final raw-pixel matrix shapes:")
    print(f"  X_BF       = {x_bf.shape}")
    print(f"  X_ED       = {x_ed.shape}")
    print(f"  X_BFplusED = {x_bfed.shape}")
    print(f"  y          = {y.shape} | classes={list(class_names)}")

    return x_bf, x_ed, x_bfed, y, list(keys)


# =============================================================================
# IO helpers
# =============================================================================
def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latex_escape(value: Any) -> str:
    s = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    return s


def write_latex_table(path: Path, rows: List[Dict[str, Any]], columns: List[str], caption: str, label: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{latex_escape(caption)}}}\n")
        f.write(f"\\label{{{latex_escape(label)}}}\n")
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(latex_escape(c) for c in columns) + " \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(" & ".join(latex_escape(row.get(c, "")) for c in columns) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def fmt_mean_std(mean_value: float, std_value: float, digits: int = 4) -> str:
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


# =============================================================================
# Metrics and figures
# =============================================================================
def per_class_accuracy_from_cm(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1)
    out = np.divide(np.diag(cm), row_sum, out=np.zeros_like(row_sum, dtype=np.float64), where=row_sum != 0)
    return out.astype(np.float64)


def collect_fold_metrics(
    mode: str,
    model_name: str,
    fold_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    train_n: int,
    test_n: int,
    dim: int,
    fit_time_s: float,
    infer_ms_per_sample: float,
    scaler_used: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_acc = per_class_accuracy_from_cm(cm)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    fold_row: Dict[str, Any] = {
        "Mode": mode,
        "Model": model_name,
        "Fold": fold_id,
        "TrainN": train_n,
        "TestN": test_n,
        "Dim": dim,
        "Accuracy": round(float(acc), 6),
        "Accuracy(%)": round(float(acc) * 100.0, 4),
        "MacroF1": round(float(macro_f1), 6),
        "WeightedF1": round(float(weighted_f1), 6),
        "FitTime(s)": round(float(fit_time_s), 6),
        "Infer(ms/sample)": round(float(infer_ms_per_sample), 6),
        "Scaler": "StandardScaler" if scaler_used else "None",
    }

    per_class_rows: List[Dict[str, Any]] = []
    for idx, cls_name in enumerate(class_names):
        cls_report = report[cls_name]
        cls_acc = float(class_acc[idx])
        cls_f1 = float(cls_report["f1-score"])
        fold_row[f"{cls_name}_ClassAcc"] = round(cls_acc, 6)
        fold_row[f"{cls_name}_ClassAcc(%)"] = round(cls_acc * 100.0, 4)
        fold_row[f"{cls_name}_F1"] = round(cls_f1, 6)

        per_class_rows.append(
            {
                "Mode": mode,
                "Model": model_name,
                "Fold": fold_id,
                "Class": cls_name,
                "ClassAcc": round(cls_acc, 6),
                "ClassAcc(%)": round(cls_acc * 100.0, 4),
                "Precision": round(float(cls_report["precision"]), 6),
                "Recall": round(float(cls_report["recall"]), 6),
                "F1": round(cls_f1, 6),
                "Support": int(cls_report["support"]),
            }
        )

    return fold_row, per_class_rows, cm, report


def save_confusion_matrix_png(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_png: Path,
    title: str,
    normalize_rows: bool,
) -> None:
    if not HAS_MPL or plt is None:
        return
    ensure_dir(out_png.parent)
    plot_cm = cm.astype(np.float64)
    if normalize_rows:
        row_sum = plot_cm.sum(axis=1, keepdims=True)
        plot_cm = np.divide(plot_cm, row_sum, out=np.zeros_like(plot_cm), where=row_sum != 0) * 100.0

    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(plot_cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            value = f"{plot_cm[i, j]:.1f}" if normalize_rows else f"{int(plot_cm[i, j])}"
            ax.text(j, i, value, ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_metric_bar(rows: List[Dict[str, Any]], out_png: Path, metric_col: str, title: str) -> None:
    if not HAS_MPL or plt is None or not rows:
        return
    ensure_dir(out_png.parent)
    labels = [str(row["Mode"]) for row in rows]
    values = [float(row[metric_col]) for row in rows]

    fig = plt.figure(figsize=(6.0, 4.0))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Mode")
    ax.set_ylabel(metric_col)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# =============================================================================
# Models
# =============================================================================
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
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def should_skip_model(model_name: str, n_train: int, dim: int) -> Optional[str]:
    """Conservative skip rules to avoid raw-pixel OOM or extreme runtime."""
    if model_name in {"KNN", "SVC_RBF"} and (dim > 60_000 or n_train > 5_000):
        return f"too slow for raw pixels (dim={dim}, train={n_train})"
    if model_name in {"LogisticRegression", "SklearnMLP"} and dim > 500_000:
        return f"too large for this model (dim={dim})"
    if model_name in {"GradientBoosting"} and dim > 300_000:
        return f"too large for GradientBoosting (dim={dim})"
    if model_name == "GaussianNB" and dim > 1_000_000:
        return f"too large for GaussianNB (dim={dim})"
    if model_name == "LightGBM" and lgb is None:
        return "lightgbm not installed"
    return None


def build_base_model(model_name: str, seed: int, n_estimators: int, max_iter: int) -> Any:
    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced")
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced")
    if model_name == "LinearSVC":
        return LinearSVC(C=1.0, max_iter=max_iter, random_state=seed, class_weight="balanced")
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
        return GradientBoostingClassifier(n_estimators=max(50, n_estimators // 3), random_state=seed)
    if model_name == "GaussianNB":
        return GaussianNB()
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if model_name == "SVC_RBF":
        return SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
    if model_name == "SklearnMLP":
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=max(80, max_iter // 10), random_state=seed)
    if model_name == "LightGBM":
        assert lgb is not None
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
    raise ValueError(f"Unknown model name: {model_name}")


def needs_scaler(model_name: str) -> bool:
    return model_name in {"LinearSVC", "SGD_LogLoss", "LogisticRegression", "KNN", "SVC_RBF", "SklearnMLP"}


def build_model_pipeline(model_name: str, seed: int, n_estimators: int, max_iter: int, dim: int, scale_dim_limit: int) -> Tuple[Any, bool]:
    base = build_base_model(model_name, seed=seed, n_estimators=n_estimators, max_iter=max_iter)
    use_scaler = needs_scaler(model_name) and dim <= scale_dim_limit
    if use_scaler:
        # The scaler is fit only inside each train fold by Pipeline.fit().
        return Pipeline([("scaler", StandardScaler()), ("clf", base)]), True
    return base, False


# =============================================================================
# Cross-validation runner
# =============================================================================
def make_cv_splits(
    y: np.ndarray,
    keys: Sequence[str],
    n_splits: int,
    seed: int,
    cv_type: str,
    eval_mode: str,
    test_size: Optional[float],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build evaluation splits.

    eval_mode="kfold":
        True StratifiedKFold / StratifiedGroupKFold. Test fraction is fixed at 1/n_splits.
        For --cv 3, each fold uses about 33.3% test data.

    eval_mode="shuffle":
        Repeated train/test splits. This is the correct mode when you want --test-size 0.3,
        0.5, or 0.7 while still repeating 3 times and reporting mean ± std.

    Note:
        sklearn has StratifiedShuffleSplit and GroupShuffleSplit, but not a built-in
        StratifiedGroupShuffleSplit. Therefore, --cv-type group + --eval-mode shuffle
        uses GroupShuffleSplit to avoid group leakage, but exact class balance may be weaker.
    """
    if n_splits < 2:
        raise RuntimeError("--cv must be at least 2.")

    if eval_mode == "shuffle":
        if test_size is None:
            test_size = 1.0 / float(n_splits)
        if not (0.0 < float(test_size) < 1.0):
            raise RuntimeError(f"--test-size must be between 0 and 1, got {test_size}")

        if cv_type == "group":
            groups = np.asarray([group_key_from_pair_key(k) for k in keys], dtype=object)
            splitter = GroupShuffleSplit(n_splits=n_splits, test_size=float(test_size), random_state=seed)
            print(
                "[INFO] Using GroupShuffleSplit. This prevents group leakage, "
                "but may not perfectly stratify classes."
            )
            return [(tr, te) for tr, te in splitter.split(np.zeros_like(y), y, groups=groups)]

        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=float(test_size), random_state=seed)
        return [(tr, te) for tr, te in splitter.split(np.zeros_like(y), y)]

    if eval_mode != "kfold":
        raise RuntimeError(f"Unknown eval_mode={eval_mode}")

    min_class_count = int(np.min(np.bincount(y)))
    if min_class_count < n_splits:
        raise RuntimeError(
            f"Cannot run {n_splits}-fold CV: smallest class has only {min_class_count} samples. "
            f"Use --cv {min_class_count} or collect more samples."
        )

    if cv_type == "group":
        if StratifiedGroupKFold is None:
            raise RuntimeError(
                "--cv-type group requires StratifiedGroupKFold (scikit-learn >= 1.1). "
                "Refusing ungrouped StratifiedKFold fallback to prevent train/test group leakage."
            )
        groups = np.asarray([group_key_from_pair_key(k) for k in keys], dtype=object)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in splitter.split(np.zeros_like(y), y, groups=groups)]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.zeros_like(y), y)]

def aggregate_results(
    fold_rows: List[Dict[str, Any]],
    perclass_rows: List[Dict[str, Any]],
    class_names: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_mode_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        by_mode_model[(str(row["Mode"]), str(row["Model"]))].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (mode, model), rows in sorted(by_mode_model.items()):
        acc_mean, acc_std = mean_std([float(r["Accuracy"]) for r in rows])
        macro_mean, macro_std = mean_std([float(r["MacroF1"]) for r in rows])
        weighted_mean, weighted_std = mean_std([float(r["WeightedF1"]) for r in rows])
        fit_mean, fit_std = mean_std([float(r["FitTime(s)"]) for r in rows])
        infer_mean, infer_std = mean_std([float(r["Infer(ms/sample)"]) for r in rows])

        out: Dict[str, Any] = {
            "Mode": mode,
            "Model": model,
            "CV": len(rows),
            "Accuracy_mean": round(acc_mean, 6),
            "Accuracy_std": round(acc_std, 6),
            "Accuracy_mean(%)": round(acc_mean * 100.0, 4),
            "Accuracy_std(%)": round(acc_std * 100.0, 4),
            "Accuracy_mean±std": fmt_mean_std(acc_mean, acc_std),
            "MacroF1_mean": round(macro_mean, 6),
            "MacroF1_std": round(macro_std, 6),
            "MacroF1_mean±std": fmt_mean_std(macro_mean, macro_std),
            "WeightedF1_mean": round(weighted_mean, 6),
            "WeightedF1_std": round(weighted_std, 6),
            "FitTime_mean(s)": round(fit_mean, 6),
            "FitTime_std(s)": round(fit_std, 6),
            "Infer_mean(ms/sample)": round(infer_mean, 6),
            "Infer_std(ms/sample)": round(infer_std, 6),
        }

        for cls_name in class_names:
            cls_acc_mean, cls_acc_std = mean_std([float(r[f"{cls_name}_ClassAcc"]) for r in rows])
            cls_f1_mean, cls_f1_std = mean_std([float(r[f"{cls_name}_F1"]) for r in rows])
            out[f"{cls_name}_ClassAcc_mean"] = round(cls_acc_mean, 6)
            out[f"{cls_name}_ClassAcc_std"] = round(cls_acc_std, 6)
            out[f"{cls_name}_ClassAcc_mean(%)"] = round(cls_acc_mean * 100.0, 4)
            out[f"{cls_name}_ClassAcc_std(%)"] = round(cls_acc_std * 100.0, 4)
            out[f"{cls_name}_ClassAcc_mean±std"] = fmt_mean_std(cls_acc_mean, cls_acc_std)
            out[f"{cls_name}_F1_mean"] = round(cls_f1_mean, 6)
            out[f"{cls_name}_F1_std"] = round(cls_f1_std, 6)
            out[f"{cls_name}_F1_mean±std"] = fmt_mean_std(cls_f1_mean, cls_f1_std)

        summary_rows.append(out)

    by_mode_model_class: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in perclass_rows:
        by_mode_model_class[(str(row["Mode"]), str(row["Model"]), str(row["Class"]))].append(row)

    perclass_summary_rows: List[Dict[str, Any]] = []
    for (mode, model, cls_name), rows in sorted(by_mode_model_class.items()):
        acc_mean, acc_std = mean_std([float(r["ClassAcc"]) for r in rows])
        p_mean, p_std = mean_std([float(r["Precision"]) for r in rows])
        r_mean, r_std = mean_std([float(r["Recall"]) for r in rows])
        f1_mean, f1_std = mean_std([float(r["F1"]) for r in rows])
        support_total = sum(int(r["Support"]) for r in rows)
        perclass_summary_rows.append(
            {
                "Mode": mode,
                "Model": model,
                "Class": cls_name,
                "ClassAcc_mean": round(acc_mean, 6),
                "ClassAcc_std": round(acc_std, 6),
                "ClassAcc_mean(%)": round(acc_mean * 100.0, 4),
                "ClassAcc_std(%)": round(acc_std * 100.0, 4),
                "Precision_mean": round(p_mean, 6),
                "Precision_std": round(p_std, 6),
                "Recall_mean": round(r_mean, 6),
                "Recall_std": round(r_std, 6),
                "F1_mean": round(f1_mean, 6),
                "F1_std": round(f1_std, 6),
                "Support_total_across_folds": support_total,
            }
        )

    return summary_rows, perclass_summary_rows


def select_best_per_mode(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_mode[str(row["Mode"])].append(row)

    best_rows: List[Dict[str, Any]] = []
    for mode, rows in sorted(by_mode.items()):
        best = sorted(
            rows,
            key=lambda r: (float(r["MacroF1_mean"]), float(r["Accuracy_mean"])),
            reverse=True,
        )[0]
        best_rows.append(best)
    return best_rows


def run_cv_for_mode(
    mode: str,
    X: np.ndarray,
    y: np.ndarray,
    keys: Sequence[str],
    class_names: Sequence[str],
    splits: List[Tuple[np.ndarray, np.ndarray]],
    model_names: Sequence[str],
    out_dir: Path,
    seed: int,
    n_estimators: int,
    max_iter: int,
    scale_dim_limit: int,
    save_fold_models: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    mode_dir = out_dir / mode
    ensure_dir(mode_dir / "reports")
    ensure_dir(mode_dir / "models")

    fold_rows: List[Dict[str, Any]] = []
    perclass_rows: List[Dict[str, Any]] = []
    cm_sum_by_model: Dict[str, np.ndarray] = {}
    n_classes = len(class_names)
    dim = int(X.shape[1])

    print("\n" + "=" * 90)
    print(f"[MODE] {mode} | samples={X.shape[0]} | dim={dim} | models={','.join(model_names)}")
    print("=" * 90)

    for model_name in model_names:
        min_train_n = min(len(train_idx) for train_idx, _ in splits)
        skip_reason = should_skip_model(model_name, n_train=int(min_train_n), dim=dim)
        if skip_reason is not None:
            print(f"[SKIP] {mode}/{model_name}: {skip_reason}")
            continue

        cm_sum_by_model[model_name] = np.zeros((n_classes, n_classes), dtype=np.int64)

        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            print(f"[CV] {mode:8s} | {model_name:20s} | fold {fold_id}/{len(splits)}")
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            model, scaler_used = build_model_pipeline(
                model_name,
                seed=seed + fold_id,
                n_estimators=n_estimators,
                max_iter=max_iter,
                dim=dim,
                scale_dim_limit=scale_dim_limit,
            )

            try:
                t0 = time.perf_counter()
                model.fit(X_train, y_train)
                fit_time_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                y_pred = model.predict(X_test)
                infer_time_s = time.perf_counter() - t0
                infer_ms_per_sample = infer_time_s / max(1, len(test_idx)) * 1000.0

                fold_row, per_cls, cm, report = collect_fold_metrics(
                    mode=mode,
                    model_name=model_name,
                    fold_id=fold_id,
                    y_true=y_test,
                    y_pred=y_pred,
                    class_names=class_names,
                    train_n=len(train_idx),
                    test_n=len(test_idx),
                    dim=dim,
                    fit_time_s=fit_time_s,
                    infer_ms_per_sample=infer_ms_per_sample,
                    scaler_used=scaler_used,
                )
                fold_rows.append(fold_row)
                perclass_rows.extend(per_cls)
                cm_sum_by_model[model_name] += cm

                print(
                    f"     Acc={fold_row['Accuracy']:.4f} | MacroF1={fold_row['MacroF1']:.4f} | "
                    + " | ".join(
                        f"{cls}_Acc={fold_row[f'{cls}_ClassAcc']:.4f}" for cls in class_names
                    )
                )

                report_path = mode_dir / "reports" / f"{model_name}_fold{fold_id}.json"
                with report_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "mode": mode,
                            "model": model_name,
                            "fold": fold_id,
                            "class_names": list(class_names),
                            "fold_metrics": fold_row,
                            "per_class_metrics": per_cls,
                            "confusion_matrix": cm.tolist(),
                            "classification_report": report,
                            "train_keys": [str(keys[i]) for i in train_idx],
                            "test_keys": [str(keys[i]) for i in test_idx],
                        },
                        f,
                        indent=2,
                    )

                if save_fold_models:
                    model_path = mode_dir / "models" / f"{model_name}_fold{fold_id}.joblib"
                    joblib.dump(model, model_path)

            except Exception as exc:
                print(f"[FAIL] {mode}/{model_name}/fold{fold_id}: {exc}")

    return fold_rows, perclass_rows, cm_sum_by_model


def train_and_save_final_best_models(
    best_rows: List[Dict[str, Any]],
    mode_to_X: Dict[str, np.ndarray],
    y: np.ndarray,
    out_dir: Path,
    seed: int,
    n_estimators: int,
    max_iter: int,
    scale_dim_limit: int,
) -> None:
    ensure_dir(out_dir / "final_models")
    for row in best_rows:
        mode = str(row["Mode"])
        model_name = str(row["Model"])
        X = mode_to_X[mode]
        dim = int(X.shape[1])
        model, _ = build_model_pipeline(
            model_name,
            seed=seed,
            n_estimators=n_estimators,
            max_iter=max_iter,
            dim=dim,
            scale_dim_limit=scale_dim_limit,
        )
        print(f"[FINAL] Training final best model on all data: {mode}/{model_name}")
        model.fit(X, y)
        model_path = out_dir / "final_models" / f"FINAL_{mode}_{model_name}.joblib"
        joblib.dump(model, model_path)
        with (out_dir / "final_models" / f"FINAL_{mode}_{model_name}.json").open("w", encoding="utf-8") as f:
            json.dump({"mode": mode, "model": model_name, "saved_model": str(model_path.name)}, f, indent=2)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="bfed_rawpix_3fold_cv_out")
    parser.add_argument("--stride", type=int, default=16, help="Raw-pixel stride. Larger is faster/smaller, e.g., 16 or 32.")
    parser.add_argument("--cv", type=int, default=3, help="Number of folds/repeated splits. Default: 3.")
    parser.add_argument("--eval-mode", type=str, default="kfold", choices=["kfold", "shuffle"], help="kfold = true CV; shuffle = repeated stratified train/test splits. Use shuffle for custom --test-size.")
    parser.add_argument("--test-size", type=float, default=None, help="Only for --eval-mode shuffle. Example: 0.3, 0.5, 0.7. If provided with kfold, eval-mode is automatically changed to shuffle.")
    parser.add_argument("--cv-type", type=str, default="stratified", choices=["stratified", "group"], help="Use group to keep duplicate sample keys in same fold.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", type=str, default="BFplusED,BF,ED")
    parser.add_argument("--models", type=str, default="fast", help="fast, all/default, or comma list.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-iter", type=int, default=8000)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--ed-norm", type=str, default="none", choices=["none", "dtype", "minmax"])
    parser.add_argument("--scale-dim-limit", type=int, default=400_000)
    parser.add_argument("--save-fold-models", action="store_true", help="Save every fold model. Can use a lot of disk.")
    parser.add_argument("--no-final-model", action="store_true", help="Do not train final best model on all data.")
    args = parser.parse_args()

    eval_mode_effective = args.eval_mode
    if args.test_size is not None and args.eval_mode == "kfold":
        print("[INFO] --test-size was provided, so switching from true kfold to repeated stratified shuffle splits.")
        eval_mode_effective = "shuffle"

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    class_names = CLASS_ORDER[:]
    label_map = {cls: idx for idx, cls in enumerate(class_names)}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not all(m in {"BFplusED", "BF", "ED"} for m in modes):
        raise ValueError("--modes must contain only BFplusED, BF, ED")
    model_names = parse_models(args.models)

    x_bf, x_ed, x_bfed, y, keys = build_paired_dataset(
        class_names=class_names,
        recursive=args.recursive,
        max_per_class=args.max_per_class,
        stride=args.stride,
        ed_norm=args.ed_norm,
    )

    mode_to_X = {
        "BF": x_bf,
        "ED": x_ed,
        "BFplusED": x_bfed,
    }

    splits = make_cv_splits(
        y=y,
        keys=keys,
        n_splits=args.cv,
        seed=args.seed,
        cv_type=args.cv_type,
        eval_mode=eval_mode_effective,
        test_size=args.test_size,
    )

    # Save meta and dataset summary.
    meta = {
        "created": now_str(),
        "task": f"classify one from {class_names}",
        "class_names": class_names,
        "label_map": label_map,
        "n_samples": int(len(y)),
        "stride": int(args.stride),
        "ed_norm": args.ed_norm,
        "cv": int(args.cv),
        "eval_mode": eval_mode_effective,
        "test_size": None if args.test_size is None else float(args.test_size),
        "cv_type": args.cv_type,
        "seed": int(args.seed),
        "modes": modes,
        "models": model_names,
        "dim_bf": int(x_bf.shape[1]),
        "dim_ed": int(x_ed.shape[1]),
        "dim_bfplused": int(x_bfed.shape[1]),
        "scale_dim_limit": int(args.scale_dim_limit),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    dataset_rows: List[Dict[str, Any]] = []
    for cls, idx in label_map.items():
        dataset_rows.append({"Class": cls, "Total": int(np.sum(y == idx))})
    dataset_rows.append({"Class": "ALL", "Total": int(len(y))})
    write_csv(out_dir / "TABLE1_dataset_summary.csv", dataset_rows, ["Class", "Total"])
    write_latex_table(
        out_dir / "TABLE1_dataset_summary.tex",
        dataset_rows,
        ["Class", "Total"],
        "Dataset summary for paired BF/ED raw-pixel samples.",
        "tab:dataset_summary",
    )

    all_fold_rows: List[Dict[str, Any]] = []
    all_perclass_rows: List[Dict[str, Any]] = []
    cm_sum_all: Dict[Tuple[str, str], np.ndarray] = {}

    print(f"[INFO] Evaluation mode: {eval_mode_effective}; cv/repeats={args.cv}; test_size={args.test_size}")

    for mode in modes:
        fold_rows, perclass_rows, cm_sum_by_model = run_cv_for_mode(
            mode=mode,
            X=mode_to_X[mode],
            y=y,
            keys=keys,
            class_names=class_names,
            splits=splits,
            model_names=model_names,
            out_dir=out_dir,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_iter=args.max_iter,
            scale_dim_limit=args.scale_dim_limit,
            save_fold_models=args.save_fold_models,
        )
        all_fold_rows.extend(fold_rows)
        all_perclass_rows.extend(perclass_rows)
        for model_name, cm_sum in cm_sum_by_model.items():
            cm_sum_all[(mode, model_name)] = cm_sum

    # Fold-level outputs.
    fold_columns = [
        "Mode",
        "Model",
        "Fold",
        "TrainN",
        "TestN",
        "Dim",
        "Accuracy",
        "Accuracy(%)",
        "MacroF1",
        "WeightedF1",
        "FitTime(s)",
        "Infer(ms/sample)",
        "Scaler",
    ]
    for cls in class_names:
        fold_columns += [f"{cls}_ClassAcc", f"{cls}_ClassAcc(%)", f"{cls}_F1"]
    write_csv(out_dir / "cv_fold_results_long.csv", all_fold_rows, fold_columns)

    perclass_columns = [
        "Mode",
        "Model",
        "Fold",
        "Class",
        "ClassAcc",
        "ClassAcc(%)",
        "Precision",
        "Recall",
        "F1",
        "Support",
    ]
    write_csv(out_dir / "cv_per_class_results_long.csv", all_perclass_rows, perclass_columns)

    summary_rows, perclass_summary_rows = aggregate_results(all_fold_rows, all_perclass_rows, class_names)

    summary_columns = [
        "Mode",
        "Model",
        "CV",
        "Accuracy_mean",
        "Accuracy_std",
        "Accuracy_mean(%)",
        "Accuracy_std(%)",
        "Accuracy_mean±std",
        "MacroF1_mean",
        "MacroF1_std",
        "MacroF1_mean±std",
        "WeightedF1_mean",
        "WeightedF1_std",
        "FitTime_mean(s)",
        "FitTime_std(s)",
        "Infer_mean(ms/sample)",
        "Infer_std(ms/sample)",
    ]
    for cls in class_names:
        summary_columns += [
            f"{cls}_ClassAcc_mean",
            f"{cls}_ClassAcc_std",
            f"{cls}_ClassAcc_mean(%)",
            f"{cls}_ClassAcc_std(%)",
            f"{cls}_ClassAcc_mean±std",
            f"{cls}_F1_mean",
            f"{cls}_F1_std",
            f"{cls}_F1_mean±std",
        ]

    write_csv(out_dir / "TABLE2_cv_summary_by_mode_model.csv", summary_rows, summary_columns)
    write_latex_table(
        out_dir / "TABLE2_cv_summary_by_mode_model.tex",
        summary_rows,
        summary_columns,
        "Cross-validation / repeated-split summary by input mode and model.",
        "tab:cv_summary",
    )

    perclass_summary_columns = [
        "Mode",
        "Model",
        "Class",
        "ClassAcc_mean",
        "ClassAcc_std",
        "ClassAcc_mean(%)",
        "ClassAcc_std(%)",
        "Precision_mean",
        "Precision_std",
        "Recall_mean",
        "Recall_std",
        "F1_mean",
        "F1_std",
        "Support_total_across_folds",
    ]
    write_csv(out_dir / "TABLE2b_cv_per_class_summary.csv", perclass_summary_rows, perclass_summary_columns)

    best_rows = select_best_per_mode(summary_rows)
    best_columns = [
        "Mode",
        "Model",
        "CV",
        "Accuracy_mean±std",
        "MacroF1_mean±std",
    ]
    for cls in class_names:
        best_columns += [f"{cls}_ClassAcc_mean±std", f"{cls}_F1_mean±std"]

    write_csv(out_dir / "TABLE3_best_model_per_mode_easy.csv", best_rows, best_columns)
    write_latex_table(
        out_dir / "TABLE3_best_model_per_mode_easy.tex",
        best_rows,
        best_columns,
        "Best model per input mode selected by mean Macro-F1 across repeated evaluation splits.",
        "tab:best_model_per_mode",
    )

    # Best model per-class table, easier for papers.
    best_keys = {(str(row["Mode"]), str(row["Model"])) for row in best_rows}
    best_perclass_rows = [
        row for row in perclass_summary_rows if (str(row["Mode"]), str(row["Model"])) in best_keys
    ]
    write_csv(out_dir / "TABLE4_best_model_per_class.csv", best_perclass_rows, perclass_summary_columns)
    write_latex_table(
        out_dir / "TABLE4_best_model_per_class.tex",
        best_perclass_rows,
        perclass_summary_columns,
        "Per-class accuracy and F1 for the best model in each input mode.",
        "tab:best_model_per_class",
    )

    # Save confusion matrices and figures for best model in each mode.
    figures_dir = out_dir / "figures"
    ensure_dir(figures_dir)
    for row in best_rows:
        mode = str(row["Mode"])
        model_name = str(row["Model"])
        cm_sum = cm_sum_all.get((mode, model_name))
        if cm_sum is None:
            continue
        np.save(figures_dir / f"CM_sum_{mode}_{model_name}.npy", cm_sum)
        save_confusion_matrix_png(
            cm_sum,
            class_names,
            figures_dir / f"FIG_CM_sum_{mode}_{model_name}_raw.png",
            title=f"{mode} / {model_name} / summed CM",
            normalize_rows=False,
        )
        save_confusion_matrix_png(
            cm_sum,
            class_names,
            figures_dir / f"FIG_CM_sum_{mode}_{model_name}_rowpct.png",
            title=f"{mode} / {model_name} / row-normalized (%)",
            normalize_rows=True,
        )

    save_metric_bar(best_rows, figures_dir / "FIG_best_MacroF1_per_mode.png", "MacroF1_mean", "Best Macro-F1 per Mode")
    save_metric_bar(best_rows, figures_dir / "FIG_best_Accuracy_per_mode.png", "Accuracy_mean", "Best Accuracy per Mode")

    if not args.no_final_model:
        train_and_save_final_best_models(
            best_rows=best_rows,
            mode_to_X=mode_to_X,
            y=y,
            out_dir=out_dir,
            seed=args.seed,
            n_estimators=args.n_estimators,
            max_iter=args.max_iter,
            scale_dim_limit=args.scale_dim_limit,
        )

    print("\n[DONE] Raw-pixel classification evaluation complete.")
    print(f"Root output: {out_dir}")
    print("Most useful files:")
    print(f"  {out_dir / 'TABLE3_best_model_per_mode_easy.csv'}")
    print(f"  {out_dir / 'TABLE4_best_model_per_class.csv'}")
    print(f"  {out_dir / 'cv_fold_results_long.csv'}")
    print(f"  {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
