#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared evaluation utilities for subgroup-safe cancer angle classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold  # type: ignore
except Exception:
    StratifiedGroupKFold = None

CLASS_ORDER = ["HPNE", "PANC", "MIA"]
N_CLASSES = len(CLASS_ORDER)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def clean_value(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        if np.isnan(v) or np.isinf(v):
            return ""
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def class_counts(y: np.ndarray) -> Dict[str, int]:
    return {cls: int(np.sum(y == i)) for i, cls in enumerate(CLASS_ORDER)}


def majority_label(y: np.ndarray) -> int:
    vals, counts = np.unique(y, return_counts=True)
    return int(vals[np.argmax(counts)])


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    denom = np.sum(e, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return e / denom


def maybe_fit_with_weight(model: Any, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> None:
    """Fit model. If estimator does not accept sample_weight, fall back safely."""
    if sample_weight is None:
        model.fit(X, y)
        return

    if hasattr(model, "steps"):
        try:
            model.fit(X, y, clf__sample_weight=sample_weight)
            return
        except TypeError:
            model.fit(X, y)
            return
        except ValueError as e:
            if "sample_weight" in str(e):
                model.fit(X, y)
                return
            raise

    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    except ValueError as e:
        if "sample_weight" in str(e):
            model.fit(X, y)
        else:
            raise


def subgroup_balanced_weights(groups: np.ndarray) -> np.ndarray:
    """Each subgroup contributes about total weight 1, regardless of image count."""
    groups = groups.astype(str)
    unique, counts = np.unique(groups, return_counts=True)
    count_map = {g: c for g, c in zip(unique, counts)}
    w = np.asarray([1.0 / float(count_map[g]) for g in groups], dtype=np.float64)
    w *= len(w) / np.sum(w)
    return w


def cv_splits_group(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(y))
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"n_groups={n_groups} is smaller than n_splits={n_splits}")
    if StratifiedGroupKFold is not None:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from sgkf.split(idx, y, groups=groups)
    else:
        print("WARNING: StratifiedGroupKFold unavailable; using GroupKFold without class stratification.")
        gkf = GroupKFold(n_splits=n_splits)
        yield from gkf.split(idx, y, groups=groups)


def stratified_group_holdout(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Class-balanced subgroup holdout. Each group is assigned its majority class.
    Chooses about test_size of groups from each class for the final test set.
    """
    rng = np.random.default_rng(seed)
    groups = groups.astype(str)
    group_to_label: Dict[str, int] = {}
    for g in np.unique(groups):
        group_to_label[g] = majority_label(y[groups == g])

    test_groups: List[str] = []
    split_info: Dict[str, Any] = {"test_size": test_size, "seed": seed, "per_class": {}}

    for cls_id, cls_name in enumerate(CLASS_ORDER):
        cls_groups = [g for g, lab in group_to_label.items() if lab == cls_id]
        cls_groups = sorted(cls_groups)
        rng.shuffle(cls_groups)
        n_total = len(cls_groups)
        n_test = int(round(n_total * test_size))
        n_test = max(1, n_test)
        n_test = min(n_test, n_total - 1) if n_total > 1 else n_total
        chosen = cls_groups[:n_test]
        test_groups.extend(chosen)
        split_info["per_class"][cls_name] = {
            "n_groups_total": n_total,
            "n_groups_test": len(chosen),
            "n_groups_train": n_total - len(chosen),
        }

    test_group_set = set(test_groups)
    test_idx = np.asarray([i for i, g in enumerate(groups) if g in test_group_set], dtype=np.int64)
    train_idx = np.asarray([i for i, g in enumerate(groups) if g not in test_group_set], dtype=np.int64)

    split_info["train_images"] = int(len(train_idx))
    split_info["test_images"] = int(len(test_idx))
    split_info["train_groups"] = int(len(np.unique(groups[train_idx])))
    split_info["test_groups"] = int(len(np.unique(groups[test_idx])))
    split_info["train_class_counts"] = class_counts(y[train_idx])
    split_info["test_class_counts"] = class_counts(y[test_idx])
    return train_idx, test_idx, split_info


def metrics_from_pred(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, Any]:
    labels = list(range(N_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    out: Dict[str, Any] = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{prefix}_confusion_matrix_json": json.dumps(cm.tolist()),
    }
    for idx, cls in enumerate(CLASS_ORDER):
        denom = cm[idx, :].sum()
        out[f"{prefix}_{cls}_acc"] = float(cm[idx, idx] / denom) if denom else 0.0
        out[f"{prefix}_{cls}_precision"] = report[cls]["precision"]
        out[f"{prefix}_{cls}_recall"] = report[cls]["recall"]
        out[f"{prefix}_{cls}_f1"] = report[cls]["f1-score"]
        out[f"{prefix}_{cls}_support"] = int(report[cls]["support"])
    return {k: clean_value(v) for k, v in out.items()}


def subgroup_metrics_from_scores(y_true: np.ndarray, groups: np.ndarray, scores: np.ndarray, prefix: str) -> Dict[str, Any]:
    groups = groups.astype(str)
    group_true: List[int] = []
    group_pred: List[int] = []
    group_sizes: List[int] = []
    for g in sorted(np.unique(groups)):
        mask = groups == g
        group_true.append(majority_label(y_true[mask]))
        mean_score = np.mean(scores[mask], axis=0)
        group_pred.append(int(np.argmax(mean_score)))
        group_sizes.append(int(np.sum(mask)))
    out = metrics_from_pred(np.asarray(group_true), np.asarray(group_pred), prefix=prefix)
    out[f"{prefix}_n_groups"] = len(group_true)
    out[f"{prefix}_group_size_mean"] = float(np.mean(group_sizes)) if group_sizes else 0.0
    out[f"{prefix}_group_size_max"] = int(np.max(group_sizes)) if group_sizes else 0
    return out


def predict_scores_multiclass(model: Any, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X).astype(int)
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X), dtype=np.float64)
            if p.ndim == 2 and p.shape[1] == N_CLASSES:
                return y_pred, p
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            z = np.asarray(model.decision_function(X), dtype=np.float64)
            if z.ndim == 1:
                z = z.reshape(-1, 1)
            if z.shape[1] == N_CLASSES:
                return y_pred, softmax(z)
        except Exception:
            pass
    scores = np.zeros((len(y_pred), N_CLASSES), dtype=np.float64)
    scores[np.arange(len(y_pred)), y_pred] = 1.0
    return y_pred, scores


def positive_score_binary(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            p = np.asarray(model.predict_proba(X), dtype=np.float64)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return sigmoid(np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1))
        except Exception:
            pass
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


def predict_ovr(models: Sequence[Any], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.vstack([positive_score_binary(m, X) for m in models]).T
    y_pred = np.argmax(scores, axis=1).astype(int)
    return y_pred, scores


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(np.mean(vals)), float(np.std(vals, ddof=1))


def flatten_mean_std(metric_rows: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    metric_lists: Dict[str, List[float]] = {}
    for row in metric_rows:
        for k, v in row.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                metric_lists.setdefault(k, []).append(float(v))
    out: Dict[str, Any] = {f"{prefix}_folds_completed": len(metric_rows)}
    for k, vals in metric_lists.items():
        mu, sd = mean_std(vals)
        out[f"{prefix}_{k}_mean"] = clean_value(mu)
        out[f"{prefix}_{k}_std"] = clean_value(sd)
    return out
