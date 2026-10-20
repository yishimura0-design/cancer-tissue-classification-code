#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU classic ML model library for HPNE/PANC/MIA image classification.

Models:
  ExtraTrees, RandomForest, LinearSVC, SGD_LogLoss, LogisticRegression,
  DecisionTree, GradientBoosting, GaussianNB, KNN, SVC_RBF, SklearnMLP,
  LightGBM optional.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None


ALL_MODELS = [
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

FAST_MODELS = [
    "ExtraTrees",
    "RandomForest",
    "LinearSVC",
    "SGD_LogLoss",
    "DecisionTree",
]


def parse_models(models_arg: str) -> List[str]:
    """
    models_arg:
      - "all" or "default" -> all models
      - "fast" -> safe first baseline models
      - comma list -> exact subset, e.g. "RandomForest,ExtraTrees,LinearSVC"
    """
    key = models_arg.strip().lower()
    if key in {"all", "default"}:
        return list(ALL_MODELS)
    if key == "fast":
        return list(FAST_MODELS)
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def should_skip_model(model_name: str, n_train: int, dim: int) -> Optional[str]:
    """Skip very slow/high-memory combinations for raw-pixel features."""
    if model_name in {"KNN", "SVC_RBF"} and (dim > 60_000 or n_train > 5_000):
        return f"too slow for raw pixels: dim={dim}, train={n_train}"
    if model_name in {"LogisticRegression", "SklearnMLP"} and dim > 500_000:
        return f"too large for this model: dim={dim}"
    if model_name == "GradientBoosting" and dim > 300_000:
        return f"too large for GradientBoosting: dim={dim}"
    if model_name == "GaussianNB" and dim > 1_000_000:
        return f"too large for GaussianNB: dim={dim}"
    if model_name == "LightGBM" and lgb is None:
        return "lightgbm not installed"
    return None


def build_base_model(
    model_name: str,
    seed: int = 42,
    n_estimators: int = 300,
    max_iter: int = 8000,
) -> Any:
    """Build one classifier with fixed settings."""
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
        return DecisionTreeClassifier(
            random_state=seed,
            class_weight="balanced",
        )

    if model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=max(50, n_estimators // 3),
            random_state=seed,
        )

    if model_name == "GaussianNB":
        return GaussianNB()

    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        )

    if model_name == "SVC_RBF":
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=False,
        )

    if model_name == "SklearnMLP":
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=max(80, max_iter // 10),
            random_state=seed,
        )

    if model_name == "LightGBM":
        if lgb is None:
            raise RuntimeError("LightGBM is not installed. Run: pip install lightgbm")
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def needs_scaler(model_name: str) -> bool:
    """Tree models do not need scaling. Linear/SVM/KNN/MLP models do."""
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
    seed: int = 42,
    n_estimators: int = 300,
    max_iter: int = 8000,
    dim: int = 0,
    scale_dim_limit: int = 400_000,
) -> Tuple[Any, bool]:
    """
    Final model builder.

    If model needs scaling and feature dimension <= scale_dim_limit:
      Pipeline(StandardScaler + classifier)
    Otherwise:
      classifier only
    """
    base_model = build_base_model(
        model_name=model_name,
        seed=seed,
        n_estimators=n_estimators,
        max_iter=max_iter,
    )

    use_scaler = needs_scaler(model_name) and dim <= scale_dim_limit
    if use_scaler:
        return Pipeline([("scaler", StandardScaler()), ("clf", base_model)]), True
    return base_model, False
