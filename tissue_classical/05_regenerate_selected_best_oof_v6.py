#!/usr/bin/env python3
"""
Exact selected-model OOF regeneration for the cancer-tissue classical-ML study.

Runs only:
  scenario1 / hist_rgb32 / ExtraTrees
  scenario2 / hist_rgb32 / LinearSVC

Outputs:
- 16,200 pooled held-out image predictions per scenario
- 225 pooled held-out subgroup predictions per scenario
- authoritative validation-only subgroup-to-fold assignment
- full train/validation subgroup assignment for leakage auditing
- image- and subgroup-level balanced accuracy by fold
- pooled subgroup confusion matrices
- pooled subgroup class-wise precision, recall/sensitivity, F1 and support
- comparison against the original fold and summary metrics

The script imports the current corrected training script and reuses its feature,
model, weighting and subgroup-aggregation functions. It deliberately fails
rather than silently changing the aggregation method or run parameters.
"""

from __future__ import annotations
import os

import argparse
import hashlib
import importlib.util
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold


DEFAULT_ROOT = Path(os.environ.get("CANCER_DATA_ROOT", "./private_data"))
FEATURE = "hist_rgb32"

SELECTED = {
    "scenario1": "ExtraTrees",
    "scenario2": "LinearSVC",
}

EXPECTED_IMAGES = 16_200
EXPECTED_GROUPS = 225
EXPECTED_FOLDS = 5
EXPECTED_GROUPS_PER_FOLD = 45
EXPECTED_IMAGES_PER_FOLD = 3_240

EXPECTED_SUPPORT = {
    "scenario1": {
        "Normal": 30,
        "Malignant": 195,
    },
    "scenario2": {
        "Stage IB": 35,
        "Stage IIA": 110,
        "Stage IIB": 30,
        "Stage III": 10,
        "Stage IV": 10,
        "Normal": 30,
    },
}

ALIASES = {
    "group_id": "subgroup_id",
    "group": "subgroup_id",
    "true_label": "y_true",
    "class": "y_true",
    "predicted_label": "y_pred",
    "prediction": "y_pred",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required file missing or empty: {path}")



def _fallback_indices_from_groups(
    groups: np.ndarray,
    selected_groups: np.ndarray,
) -> np.ndarray:
    selected = set(map(str, selected_groups))
    return np.asarray(
        [index for index, group in enumerate(groups) if str(group) in selected],
        dtype=int,
    )


def _fallback_subgroup_balanced_sample_weights(
    groups_subset: np.ndarray,
) -> np.ndarray:
    values = np.asarray(groups_subset).astype(str)
    counts = pd.Series(values).value_counts().to_dict()
    return np.asarray(
        [1.0 / float(counts[group]) for group in values],
        dtype=np.float32,
    )


def _fallback_fit_with_optional_sample_weight(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    if sample_weight is None:
        model.fit(X_train, y_train)
        return

    # Pipelines conventionally route weights to the final "clf" step.
    try:
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            model.fit(
                X_train,
                y_train,
                clf__sample_weight=sample_weight,
            )
        else:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
            )
        return
    except (TypeError, ValueError) as error:
        if "sample_weight" not in str(error):
            raise

    # Preserve the training script's documented fallback for estimators that
    # reject sample weights.
    model.fit(X_train, y_train)


def _estimator_classes(model: Any) -> list[str]:
    classes = list(getattr(model, "classes_", []))
    if classes:
        return [str(value) for value in classes]

    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("clf")
        classes = list(getattr(classifier, "classes_", []))
        if classes:
            return [str(value) for value in classes]

    return []


def _fallback_predict_groups(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    class_order: list[str],
):
    """
    Exact supported aggregation:
      - average predict_proba scores when available;
      - otherwise average decision_function scores;
      - majority vote only when neither score interface exists.
    """
    groups_array = np.asarray(groups_test).astype(str)
    true_array = np.asarray(y_test).astype(str)
    image_predictions = np.asarray(model.predict(X_test)).astype(str)
    unique_groups = list(dict.fromkeys(groups_array.tolist()))
    model_classes = _estimator_classes(model)

    score_matrix = None
    method = None

    if hasattr(model, "predict_proba"):
        try:
            score_matrix = np.asarray(model.predict_proba(X_test))
            method = "avg_proba"
        except Exception:
            score_matrix = None

    if score_matrix is None and hasattr(model, "decision_function"):
        try:
            raw_scores = np.asarray(model.decision_function(X_test))
            if raw_scores.ndim == 1:
                # Binary LinearSVC returns one signed score. Convert it to two
                # ordered class scores aligned with model.classes_.
                raw_scores = np.column_stack([-raw_scores, raw_scores])
            score_matrix = raw_scores
            method = "decision_score"
        except Exception:
            score_matrix = None

    if score_matrix is not None and not model_classes:
        raise RuntimeError(
            "The model exposed class scores but no classes_ ordering; "
            "subgroup predictions cannot be safely reconstructed."
        )

    rows = []
    group_predictions = []

    for subgroup_id in unique_groups:
        mask = groups_array == subgroup_id
        true_counts = pd.Series(true_array[mask]).value_counts()
        y_true_group = str(true_counts.index[0])

        if score_matrix is not None:
            average_scores = np.asarray(score_matrix[mask]).mean(axis=0)
            class_scores = {label: float("-inf") for label in class_order}

            for index, model_class in enumerate(model_classes):
                if model_class in class_scores and index < len(average_scores):
                    class_scores[model_class] = float(average_scores[index])

            y_pred_group = max(
                class_order,
                key=lambda label: (
                    class_scores[label],
                    -class_order.index(label),
                ),
            )
            confidence = class_scores[y_pred_group]
            vote_method = method
        else:
            votes = pd.Series(image_predictions[mask]).value_counts()
            maximum = int(votes.max())
            tied = set(votes[votes == maximum].index.astype(str))
            y_pred_group = next(
                label for label in class_order if label in tied
            )
            confidence = maximum / int(mask.sum())
            vote_method = "majority_vote"

        group_predictions.append(y_pred_group)
        rows.append(
            {
                "group_id": subgroup_id,
                "y_true": y_true_group,
                "y_pred": y_pred_group,
                "correct": int(y_true_group == y_pred_group),
                "n_images": int(mask.sum()),
                "confidence_or_vote_fraction": float(confidence),
                "vote_method": vote_method,
            }
        )

    return np.asarray(group_predictions), pd.DataFrame(rows)




def _compatible_make_group_table(
    module: Any,
    dataframe: pd.DataFrame,
    label_column: str,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """
    Support both training-script interfaces observed in this project:

    Newer CCAST script:
        make_group_table(sdf, label_column)

    Older clean-package script:
        make_group_table(y, groups)

    The returned table is normalized to exactly:
        group, label
    """
    attempts = [
        (
            "make_group_table(dataframe, label_column)",
            (dataframe, label_column),
        ),
        (
            "make_group_table(dataframe)",
            (dataframe,),
        ),
        (
            "make_group_table(y, groups)",
            (y, groups),
        ),
    ]

    errors: list[str] = []

    for description, arguments in attempts:
        try:
            result = module.make_group_table(*arguments)
        except Exception as error:
            errors.append(
                f"{description}: {type(error).__name__}: {error}"
            )
            continue

        if not isinstance(result, pd.DataFrame):
            errors.append(
                f"{description}: returned {type(result).__name__}, "
                "expected pandas.DataFrame"
            )
            continue

        table = result.copy()

        if "group" not in table.columns:
            for candidate in [
                "subgroup_id",
                "group_id",
                "subgroup",
            ]:
                if candidate in table.columns:
                    table = table.rename(
                        columns={candidate: "group"}
                    )
                    break

        if "label" not in table.columns:
            for candidate in [
                label_column,
                "class",
                "target",
                "y",
            ]:
                if candidate in table.columns:
                    table = table.rename(
                        columns={candidate: "label"}
                    )
                    break

        if {"group", "label"}.issubset(table.columns):
            table = table[["group", "label"]].copy()
            table["group"] = table["group"].astype(str)
            table["label"] = table["label"].astype(str)

            if table["group"].duplicated().any():
                errors.append(
                    f"{description}: returned duplicate subgroup rows"
                )
                continue

            table = table.sort_values(
                "group",
                kind="stable",
            ).reset_index(drop=True)

            print(
                "[PASS] Group table interface: "
                f"{description}"
            )
            return table

        errors.append(
            f"{description}: columns {list(table.columns)} "
            "could not be normalized to group/label"
        )

    # Final audited fallback. The active CCAST implementation shown in the
    # traceback groups the scenario DataFrame by subgroup_id and assigns one
    # class label to each subgroup. Reproduce that behavior explicitly.
    rows = []

    for subgroup_id, group_df in dataframe.groupby(
        "subgroup_id",
        sort=True,
    ):
        labels = (
            group_df[label_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(labels) != 1:
            raise RuntimeError(
                f"Subgroup {subgroup_id} has {len(labels)} "
                f"distinct {label_column} labels: {labels}"
            )

        rows.append(
            {
                "group": str(subgroup_id),
                "label": str(labels[0]),
            }
        )

    table = pd.DataFrame(rows)

    if table.empty:
        raise RuntimeError(
            "Could not construct a subgroup table. "
            "Attempted interfaces:\n- "
            + "\n- ".join(errors)
        )

    if table["group"].duplicated().any():
        raise RuntimeError(
            "Locally constructed subgroup table contains duplicates"
        )

    print(
        "[PASS] Group table interface: audited local "
        "DataFrame grouping fallback"
    )

    return table.reset_index(drop=True)



def load_module(path: Path):
    require_file(path)
    spec = importlib.util.spec_from_file_location("ct_training_current", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = [
        "SCENARIOS",
        "infer_subgroup_id",
        "make_group_table",
        "default_feature_space",
        "feature_cache_key",
        "extract_feature_matrix",
        "build_model_pipeline",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "Training script is missing indispensable functions: "
            f"{missing}"
        )

    # The CCAST copy currently does not expose several helper functions.
    # Bind audited local equivalents rather than requiring a different
    # training script or silently changing the experiment.
    if not hasattr(module, "indices_from_groups"):
        module.indices_from_groups = _fallback_indices_from_groups
    if not hasattr(module, "subgroup_balanced_sample_weights"):
        module.subgroup_balanced_sample_weights = (
            _fallback_subgroup_balanced_sample_weights
        )
    if not hasattr(module, "fit_with_optional_sample_weight"):
        module.fit_with_optional_sample_weight = (
            _fallback_fit_with_optional_sample_weight
        )

    # Always use the local score-aware implementation. This is required for
    # the selected Scenario 2 LinearSVC result, whose subgroup aggregation is
    # recorded as decision_score.
    module.predict_groups = _fallback_predict_groups

    return module


def locate_run_settings(source_feature_root: Path, scenario: str) -> tuple[dict[str, Any], str]:
    candidates = [
        source_feature_root / "run_settings.json",
        source_feature_root / scenario / "run_settings.json",
    ]
    candidates.extend(source_feature_root.rglob("run_settings.json"))

    seen: set[str] = set()
    valid: list[tuple[Path, dict[str, Any]]] = []

    for path in candidates:
        if str(path) in seen or not path.exists():
            continue
        seen.add(str(path))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        scenarios = data.get("scenarios")
        features = data.get("features")
        scenario_ok = scenarios is None or scenario in scenarios
        feature_ok = features is None or FEATURE in features
        if scenario_ok and feature_ok:
            valid.append((path, data))

    if not valid:
        raise FileNotFoundError(
            f"No run_settings.json covering {scenario}/{FEATURE} was found under "
            f"{source_feature_root}. Exact regeneration should not use guessed parameters."
        )

    signatures: dict[str, list[str]] = {}
    for path, data in valid:
        signature_data = {
            key: data.get(key)
            for key in [
                "seed",
                "folds",
                "n_estimators",
                "max_iter",
                "scale_dim_limit",
                "manifest",
            ]
        }
        signature = json.dumps(signature_data, sort_keys=True)
        signatures.setdefault(signature, []).append(str(path))

    if len(signatures) != 1:
        raise RuntimeError(
            "Conflicting run settings were found for the selected source result:\n"
            + "\n".join(
                f"{signature}: {paths}" for signature, paths in signatures.items()
            )
        )

    path, data = valid[0]
    return data, str(path)


def prepare_manifest(module, manifest_path: Path, scenario: str) -> pd.DataFrame:
    require_file(manifest_path)
    df = pd.read_csv(manifest_path)

    for column in ["filepath", "sample_id_from_folder"]:
        if column not in df.columns:
            raise RuntimeError(f"Manifest missing required column: {column}")

    if "qc_status" in df.columns:
        df = df[df["qc_status"].astype(str).str.startswith("accepted")].copy()

    df["filepath"] = df["filepath"].astype(str)
    df["sample_id_from_folder"] = (
        df["sample_id_from_folder"].astype(str).str.upper()
    )

    if "relative_path" not in df.columns:
        df["relative_path"] = df["filepath"].map(lambda value: Path(value).name)
    if "filename" not in df.columns:
        df["filename"] = df["filepath"].map(lambda value: Path(value).name)

    df["subgroup_id"] = [
        module.infer_subgroup_id(relative_path, sample_id, filename)
        for relative_path, sample_id, filename in zip(
            df["relative_path"].astype(str),
            df["sample_id_from_folder"].astype(str),
            df["filename"].astype(str),
        )
    ]

    info = module.SCENARIOS[scenario]
    label_column = info["label_col"]
    classes = list(info["classes"])

    if label_column not in df.columns:
        raise RuntimeError(f"Manifest missing scenario label column: {label_column}")

    df = df[
        df[label_column].notna()
        & df[label_column].astype(str).isin(classes)
    ].copy()
    df = df.reset_index(drop=True)
    df["manifest_index"] = np.arange(len(df), dtype=int)

    if len(df) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"{scenario}: expected {EXPECTED_IMAGES} accepted images, found {len(df)}"
        )
    if df["subgroup_id"].nunique() != EXPECTED_GROUPS:
        raise RuntimeError(
            f"{scenario}: expected {EXPECTED_GROUPS} subgroups, found "
            f"{df['subgroup_id'].nunique()}"
        )

    missing = [value for value in df["filepath"] if not Path(value).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest image files do not exist. Example: {missing[0]}"
        )

    return df


def load_feature_matrix(
    module,
    source_feature_root: Path,
    output_root: Path,
    scenario: str,
    paths: list[str],
) -> tuple[np.ndarray, dict[str, Any], str]:
    feature_space = module.default_feature_space()
    if FEATURE not in feature_space:
        raise RuntimeError(f"{FEATURE} is absent from the current feature space")
    spec = feature_space[FEATURE]
    key = module.feature_cache_key(paths, FEATURE, spec)
    cache_name = f"features_{FEATURE}_{key}.joblib"

    candidates = [
        source_feature_root / "feature_cache" / scenario / cache_name,
        source_feature_root / "feature_cache" / cache_name,
    ]
    candidates.extend(source_feature_root.rglob(cache_name))

    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = joblib.load(candidate)
        if not isinstance(payload, dict) or "X" not in payload:
            continue
        X = np.asarray(payload["X"])
        if X.shape[0] != len(paths):
            continue
        return X, dict(payload.get("meta", {})), str(candidate)

    cache_dir = output_root / "feature_cache" / scenario
    X, metadata = module.extract_feature_matrix(
        paths=paths,
        feature_name=FEATURE,
        spec=spec,
        cache_dir=cache_dir,
        use_cache=True,
    )
    return np.asarray(X), dict(metadata), str(metadata.get("cache_path", cache_dir))


def normalize_group_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source, target in ALIASES.items():
        if source in result.columns and target not in result.columns:
            result = result.rename(columns={source: target})

    required = ["subgroup_id", "y_true", "y_pred"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise RuntimeError(
            f"predict_groups output missing {missing}; columns={list(result.columns)}"
        )

    result["subgroup_id"] = result["subgroup_id"].astype(str)
    result["y_true"] = result["y_true"].astype(str)
    result["y_pred"] = result["y_pred"].astype(str)
    return result


def metrics(y_true, y_pred, classes: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=classes,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def per_class(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": classes,
            "precision": precision,
            "recall_sensitivity": recall,
            "f1_score": f1,
            "support": support.astype(int),
        }
    )


def observed_aggregation_method(group_frame: pd.DataFrame) -> set[str]:
    columns = [
        column
        for column in [
            "vote_method",
            "aggregation_method",
            "subgroup_aggregation_method",
            "method",
        ]
        if column in group_frame.columns
    ]
    observed: set[str] = set()
    for column in columns:
        observed.update(
            group_frame[column].dropna().astype(str).str.lower().tolist()
        )
    return observed


def expected_aggregation(summary_row: pd.Series) -> str:
    for column in [
        "subgroup_aggregation_methods",
        "subgroup_aggregation_method",
        "aggregation_method",
    ]:
        if column in summary_row.index and not pd.isna(summary_row[column]):
            return str(summary_row[column]).strip().lower()
    return ""



def _canonical_aggregation_name(value: str) -> str:
    """
    Normalize equivalent aggregation labels used across historical outputs.

    Canonical values:
      probability     -> averaging class probabilities
      decision_score  -> averaging decision-function scores
      majority_vote   -> majority vote across image predictions
    """
    token = str(value).strip().lower()
    token = token.replace("-", "_").replace(" ", "_")

    aliases = {
        "probability": "probability",
        "probabilities": "probability",
        "probability_average": "probability",
        "average_probability": "probability",
        "average_probabilities": "probability",
        "avg_probability": "probability",
        "avg_probabilities": "probability",
        "avg_proba": "probability",
        "mean_probability": "probability",
        "mean_probabilities": "probability",
        "predict_proba": "probability",

        "decision_score": "decision_score",
        "decision_scores": "decision_score",
        "average_decision_score": "decision_score",
        "average_decision_scores": "decision_score",
        "avg_decision_score": "decision_score",
        "avg_decision_scores": "decision_score",
        "mean_decision_score": "decision_score",
        "mean_decision_scores": "decision_score",
        "decision_function": "decision_score",

        "majority_vote": "majority_vote",
        "majority": "majority_vote",
        "vote": "majority_vote",
        "hard_vote": "majority_vote",
    }

    return aliases.get(token, token)



def verify_aggregation(
    module,
    model_name: str,
    group_frame: pd.DataFrame,
    summary_row: pd.Series,
) -> str:
    observed = observed_aggregation_method(group_frame)
    expected = expected_aggregation(summary_row)

    expected_tokens = {
        _canonical_aggregation_name(token)
        for token in expected.replace(";", ",").split(",")
        if token.strip()
    }
    observed_tokens = {
        _canonical_aggregation_name(token)
        for token in observed
        if str(token).strip()
    }

    if expected_tokens and observed_tokens and observed_tokens != expected_tokens:
        raise RuntimeError(
            "Subgroup aggregation mismatch after alias normalization. "
            f"Original summary={expected_tokens}; "
            f"regeneration={observed_tokens}; "
            f"raw original={expected!r}; raw regenerated={observed}"
        )

    if model_name == "LinearSVC":
        source = inspect.getsource(module.predict_groups).lower()
        supports_decision = "decision_function" in source

        if "decision_score" in expected_tokens and (
            "decision_score" not in observed_tokens
        ):
            raise RuntimeError(
                "The selected LinearSVC result records decision-score subgroup "
                "aggregation, but the regeneration did not use it. "
                "Do not use majority voting for this result."
            )

        if "decision_score" in expected_tokens and not supports_decision:
            raise RuntimeError(
                "The score-aware prediction implementation does not contain "
                "decision_function handling, but the selected LinearSVC result "
                "requires decision-score subgroup aggregation."
            )

    return (
        ",".join(sorted(observed_tokens))
        if observed_tokens
        else "not_reported"
    )


def select_original_rows(
    source_feature_root: Path,
    scenario: str,
    model_name: str,
) -> tuple[pd.Series, pd.DataFrame]:
    summary_path = source_feature_root / "results_summary.csv"
    fold_path = source_feature_root / "fold_results.csv"
    require_file(summary_path)
    require_file(fold_path)

    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(fold_path)

    selected_summary = summary[
        summary["scenario"].eq(scenario)
        & summary["feature"].eq(FEATURE)
        & summary["model"].eq(model_name)
        & summary["cv_status"].astype(str).str.lower().eq("ok")
    ]
    selected_folds = folds[
        folds["scenario"].eq(scenario)
        & folds["feature"].eq(FEATURE)
        & folds["model"].eq(model_name)
    ].copy()

    if len(selected_summary) != 1:
        raise RuntimeError(
            f"Expected one source summary row for {scenario}/{FEATURE}/{model_name}; "
            f"found {len(selected_summary)}"
        )
    if len(selected_folds) != EXPECTED_FOLDS:
        raise RuntimeError(
            f"Expected five source fold rows for {scenario}/{FEATURE}/{model_name}; "
            f"found {len(selected_folds)}"
        )

    return selected_summary.iloc[0], selected_folds


def compare_metrics(
    source_summary: pd.Series,
    source_folds: pd.DataFrame,
    regenerated_folds: pd.DataFrame,
    pooled_image: dict[str, float],
    pooled_group: dict[str, float],
    classwise: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    fold_column_pairs = {
        "fold_image_acc": "image_accuracy",
        "fold_image_macro_f1": "image_macro_f1",
        "fold_image_weighted_f1": "image_weighted_f1",
        "fold_group_acc": "subgroup_accuracy",
        "fold_group_macro_f1": "subgroup_macro_f1",
        "fold_group_weighted_f1": "subgroup_weighted_f1",
    }

    source_folds = source_folds.sort_values("fold").reset_index(drop=True)
    regenerated_folds = regenerated_folds.sort_values("fold").reset_index(drop=True)

    for source_column, regenerated_column in fold_column_pairs.items():
        if source_column not in source_folds.columns:
            continue
        for index in range(EXPECTED_FOLDS):
            original = source_folds.loc[index, source_column]
            regenerated = regenerated_folds.loc[index, regenerated_column]
            difference = abs(float(original) - float(regenerated))
            rows.append(
                {
                    "scope": f"fold_{index + 1}",
                    "metric": source_column,
                    "original_value": float(original),
                    "regenerated_value": float(regenerated),
                    "absolute_difference": difference,
                    "match": difference <= tolerance,
                }
            )

    summary_values = {
        "cv_image_acc_mean": regenerated_folds["image_accuracy"].mean(),
        "cv_image_acc_std": regenerated_folds["image_accuracy"].std(ddof=1),
        "cv_image_macro_f1_mean": regenerated_folds["image_macro_f1"].mean(),
        "cv_image_macro_f1_std": regenerated_folds["image_macro_f1"].std(ddof=1),
        "cv_image_weighted_f1_mean": regenerated_folds["image_weighted_f1"].mean(),
        "cv_image_weighted_f1_std": regenerated_folds["image_weighted_f1"].std(ddof=1),
        "cv_group_acc_mean": regenerated_folds["subgroup_accuracy"].mean(),
        "cv_group_acc_std": regenerated_folds["subgroup_accuracy"].std(ddof=1),
        "cv_group_macro_f1_mean": regenerated_folds["subgroup_macro_f1"].mean(),
        "cv_group_macro_f1_std": regenerated_folds["subgroup_macro_f1"].std(ddof=1),
        "cv_group_weighted_f1_mean": regenerated_folds["subgroup_weighted_f1"].mean(),
        "cv_group_weighted_f1_std": regenerated_folds["subgroup_weighted_f1"].std(ddof=1),
        "oof_image_acc": pooled_image["accuracy"],
        "oof_image_macro_f1": pooled_image["macro_f1"],
        "oof_image_weighted_f1": pooled_image["weighted_f1"],
        "oof_group_acc": pooled_group["accuracy"],
        "oof_group_macro_f1": pooled_group["macro_f1"],
        "oof_group_weighted_f1": pooled_group["weighted_f1"],
    }

    for _, class_row in classwise.iterrows():
        key = str(class_row["class"]).replace(" ", "_").replace("/", "_")
        summary_values[f"oof_group_{key}_precision"] = class_row["precision"]
        summary_values[f"oof_group_{key}_recall"] = class_row["recall_sensitivity"]
        summary_values[f"oof_group_{key}_f1"] = class_row["f1_score"]

    for metric_name, regenerated in summary_values.items():
        if metric_name not in source_summary.index or pd.isna(source_summary[metric_name]):
            rows.append(
                {
                    "scope": "summary",
                    "metric": metric_name,
                    "original_value": np.nan,
                    "regenerated_value": float(regenerated),
                    "absolute_difference": np.nan,
                    "match": "original_not_available",
                }
            )
            continue

        original = float(source_summary[metric_name])
        difference = abs(original - float(regenerated))
        rows.append(
            {
                "scope": "summary",
                "metric": metric_name,
                "original_value": original,
                "regenerated_value": float(regenerated),
                "absolute_difference": difference,
                "match": difference <= tolerance,
            }
        )

    return pd.DataFrame(rows)


def run_scenario(args, scenario: str) -> None:
    root = args.root
    training_script = root / "cancer_tissue_clean_rerun/scripts/03_train_cancer_searchspace_grouped.py"
    manifest_default = root / "qc_outputs_clean/manifest_accepted.csv"
    source_feature_root = root / "ml_results_cancer_tissue_grouped_clean_stage1_5fold" / FEATURE
    output_root = root / "selected_best_oof_regeneration_v3"
    scenario_out = output_root / scenario
    scenario_out.mkdir(parents=True, exist_ok=True)

    module = load_module(training_script)
    settings, settings_path = locate_run_settings(source_feature_root, scenario)

    seed = int(settings.get("seed", 42))
    folds = int(settings.get("folds", 5))
    n_estimators = int(settings.get("n_estimators", 300))
    max_iter = int(settings.get("max_iter", 8000))
    scale_dim_limit = int(settings.get("scale_dim_limit", 400_000))

    if folds != EXPECTED_FOLDS:
        raise RuntimeError(f"Expected five folds, source run settings contain {folds}")

    manifest_path = Path(settings.get("manifest", manifest_default))
    if not manifest_path.exists():
        manifest_path = manifest_default

    model_name = SELECTED[scenario]
    source_summary, source_folds = select_original_rows(
        source_feature_root, scenario, model_name
    )

    if "sample_weight_used_folds" in source_summary.index:
        expected_weighted_folds = int(
            source_summary["sample_weight_used_folds"]
        )
        if expected_weighted_folds != 0:
            raise RuntimeError(
                "This regeneration expects the selected clean result to "
                "record sample_weight_used_folds=0, but found "
                f"{expected_weighted_folds}."
            )

    df = prepare_manifest(module, manifest_path, scenario)
    info = module.SCENARIOS[scenario]
    label_column = info["label_col"]
    classes = list(info["classes"])

    paths = df["filepath"].astype(str).tolist()
    y = df[label_column].astype(str).to_numpy()
    groups = df["subgroup_id"].astype(str).to_numpy()

    X, feature_metadata, cache_source = load_feature_matrix(
        module, source_feature_root, output_root, scenario, paths
    )
    if X.shape[0] != EXPECTED_IMAGES:
        raise RuntimeError(f"Feature matrix rows={X.shape[0]}, expected 16200")

    group_table = _compatible_make_group_table(
        module=module,
        dataframe=df,
        label_column=label_column,
        y=y,
        groups=groups,
    )
    if len(group_table) != EXPECTED_GROUPS:
        raise RuntimeError(f"Group table rows={len(group_table)}, expected 225")

    group_values = group_table["group"].astype(str).to_numpy()
    group_labels = group_table["label"].astype(str).to_numpy()

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    validation_assignments = []
    all_split_assignments = []
    image_oof_parts = []
    group_oof_parts = []
    fold_metric_rows = []
    aggregation_methods: set[str] = set()

    for fold_number, (train_group_indices, validation_group_indices) in enumerate(
        splitter.split(group_values, group_labels),
        start=1,
    ):
        train_groups = group_values[train_group_indices]
        validation_groups = group_values[validation_group_indices]

        overlap = set(train_groups) & set(validation_groups)
        if overlap:
            raise RuntimeError(
                f"Fold {fold_number}: train/validation subgroup overlap: "
                f"{sorted(overlap)[:10]}"
            )

        train_indices = module.indices_from_groups(groups, train_groups)
        validation_indices = module.indices_from_groups(groups, validation_groups)

        if len(validation_groups) != EXPECTED_GROUPS_PER_FOLD:
            raise RuntimeError(
                f"Fold {fold_number}: validation groups={len(validation_groups)}, expected 45"
            )
        if len(validation_indices) != EXPECTED_IMAGES_PER_FOLD:
            raise RuntimeError(
                f"Fold {fold_number}: validation images={len(validation_indices)}, expected 3240"
            )

        model, scaler_used = module.build_model_pipeline(
            model_name=model_name,
            seed=seed + fold_number,
            n_estimators=n_estimators,
            max_iter=max_iter,
            dim=X.shape[1],
            scale_dim_limit=scale_dim_limit,
        )

        # Match the clean original run exactly. The retained result records
        # sample_weight_used_folds=0 because every training subgroup contains
        # the same number of images. In that case subgroup balancing would be
        # only a constant multiplier, so no sample weights were passed.
        training_group_counts = (
            pd.Series(groups[train_indices].astype(str))
            .value_counts()
        )

        if training_group_counts.nunique() == 1:
            sample_weights = None
            sample_weight_applied = False
            sample_weight_note = (
                "not applied because all training subgroups "
                "have equal image counts"
            )
        else:
            sample_weights = module.subgroup_balanced_sample_weights(
                groups[train_indices]
            )
            sample_weight_applied = True
            sample_weight_note = (
                "subgroup-balanced sample weights applied"
            )

        module.fit_with_optional_sample_weight(
            model,
            X[train_indices],
            y[train_indices],
            sample_weights,
        )

        image_predictions = model.predict(X[validation_indices])
        _, raw_group_frame = module.predict_groups(
            model=model,
            X_test=X[validation_indices],
            y_test=y[validation_indices],
            groups_test=groups[validation_indices],
            class_order=classes,
        )
        group_frame = normalize_group_frame(raw_group_frame)
        method = verify_aggregation(module, model_name, group_frame, source_summary)
        aggregation_methods.add(method)

        expected_group_labels = dict(
            zip(
                group_values[validation_group_indices],
                group_labels[validation_group_indices],
            )
        )
        for subgroup_id, true_label in zip(
            group_frame["subgroup_id"], group_frame["y_true"]
        ):
            if expected_group_labels[subgroup_id] != true_label:
                raise RuntimeError(
                    f"Fold {fold_number}: true-label mismatch for {subgroup_id}"
                )

        image_frame = pd.DataFrame(
            {
                "scenario": scenario,
                "feature": FEATURE,
                "model": model_name,
                "fold": fold_number,
                "manifest_index": validation_indices,
                "filepath": df.iloc[validation_indices]["filepath"].astype(str).to_numpy(),
                "subgroup_id": groups[validation_indices],
                "y_true": y[validation_indices],
                "y_pred": image_predictions,
            }
        )
        image_frame["correct"] = (
            image_frame["y_true"] == image_frame["y_pred"]
        ).astype(int)

        group_frame.insert(0, "scenario", scenario)
        group_frame.insert(1, "feature", FEATURE)
        group_frame.insert(2, "model", model_name)
        group_frame.insert(3, "fold", fold_number)

        image_oof_parts.append(image_frame)
        group_oof_parts.append(group_frame)

        validation_assignment = (
            group_table.iloc[validation_group_indices][["group", "label"]]
            .rename(columns={"group": "subgroup_id", "label": "class"})
            .copy()
        )
        validation_assignment.insert(0, "scenario", scenario)
        validation_assignment["fold"] = fold_number
        validation_assignments.append(validation_assignment)

        for split_role, indices in [
            ("train", train_group_indices),
            ("validation", validation_group_indices),
        ]:
            part = (
                group_table.iloc[indices][["group", "label"]]
                .rename(columns={"group": "subgroup_id", "label": "class"})
                .copy()
            )
            part.insert(0, "scenario", scenario)
            part["fold"] = fold_number
            part["split_role"] = split_role
            all_split_assignments.append(part)

        image_metrics = metrics(
            image_frame["y_true"].to_numpy(),
            image_frame["y_pred"].to_numpy(),
            classes,
        )
        group_metrics = metrics(
            group_frame["y_true"].to_numpy(),
            group_frame["y_pred"].to_numpy(),
            classes,
        )

        fold_metric_rows.append(
            {
                "scenario": scenario,
                "feature": FEATURE,
                "model": model_name,
                "fold": fold_number,
                "train_images": len(train_indices),
                "validation_images": len(validation_indices),
                "train_subgroups": len(train_groups),
                "validation_subgroups": len(validation_groups),
                "scaler_used": int(scaler_used),
                "sample_weight_applied": int(sample_weight_applied),
                "sample_weight_note": sample_weight_note,
                "training_images_per_subgroup_min": int(
                    training_group_counts.min()
                ),
                "training_images_per_subgroup_max": int(
                    training_group_counts.max()
                ),
                "aggregation_method": method,
                "image_accuracy": image_metrics["accuracy"],
                "image_balanced_accuracy": image_metrics["balanced_accuracy"],
                "image_macro_f1": image_metrics["macro_f1"],
                "image_weighted_f1": image_metrics["weighted_f1"],
                "subgroup_accuracy": group_metrics["accuracy"],
                "subgroup_balanced_accuracy": group_metrics["balanced_accuracy"],
                "subgroup_macro_f1": group_metrics["macro_f1"],
                "subgroup_weighted_f1": group_metrics["weighted_f1"],
            }
        )

    validation_assignment = pd.concat(validation_assignments, ignore_index=True)
    split_assignment = pd.concat(all_split_assignments, ignore_index=True)
    image_oof = pd.concat(image_oof_parts, ignore_index=True)
    group_oof = pd.concat(group_oof_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    validation_assignment = validation_assignment.sort_values(
        ["fold", "class", "subgroup_id"]
    ).reset_index(drop=True)
    split_assignment = split_assignment.sort_values(
        ["fold", "split_role", "class", "subgroup_id"]
    ).reset_index(drop=True)
    image_oof = image_oof.sort_values(["fold", "manifest_index"]).reset_index(drop=True)
    group_oof = group_oof.sort_values(["fold", "subgroup_id"]).reset_index(drop=True)

    if len(validation_assignment) != EXPECTED_GROUPS:
        raise RuntimeError("Validation-only assignment must contain 225 rows")
    if validation_assignment["subgroup_id"].nunique() != EXPECTED_GROUPS:
        raise RuntimeError("Every subgroup must occur in exactly one validation fold")
    if len(group_oof) != EXPECTED_GROUPS:
        raise RuntimeError("Pooled subgroup OOF predictions must contain 225 rows")
    if group_oof["subgroup_id"].nunique() != EXPECTED_GROUPS:
        raise RuntimeError("Pooled subgroup OOF predictions contain duplicate groups")
    if len(image_oof) != EXPECTED_IMAGES:
        raise RuntimeError("Pooled image OOF predictions must contain 16200 rows")
    if image_oof["manifest_index"].nunique() != EXPECTED_IMAGES:
        raise RuntimeError("Pooled image OOF predictions contain duplicate images")

    # Explicit train-versus-validation leakage audit for every fold.
    leakage_rows = []
    for fold_number in range(1, EXPECTED_FOLDS + 1):
        fold_part = split_assignment[split_assignment["fold"] == fold_number]
        train_set = set(
            fold_part[fold_part["split_role"] == "train"]["subgroup_id"]
        )
        validation_set = set(
            fold_part[fold_part["split_role"] == "validation"]["subgroup_id"]
        )
        overlap = sorted(train_set & validation_set)
        leakage_rows.append(
            {
                "scenario": scenario,
                "fold": fold_number,
                "train_subgroups": len(train_set),
                "validation_subgroups": len(validation_set),
                "overlap_count": len(overlap),
                "overlap_examples": ";".join(overlap[:10]),
                "pass": len(overlap) == 0,
            }
        )
    leakage_audit = pd.DataFrame(leakage_rows)
    if not leakage_audit["pass"].all():
        raise RuntimeError("At least one fold contains train/validation subgroup overlap")

    pooled_image = metrics(
        image_oof["y_true"].to_numpy(),
        image_oof["y_pred"].to_numpy(),
        classes,
    )
    pooled_group = metrics(
        group_oof["y_true"].to_numpy(),
        group_oof["y_pred"].to_numpy(),
        classes,
    )

    classwise = per_class(
        group_oof["y_true"].to_numpy(),
        group_oof["y_pred"].to_numpy(),
        classes,
    )
    observed_support = dict(
        zip(classwise["class"], classwise["support"].astype(int))
    )
    if observed_support != EXPECTED_SUPPORT[scenario]:
        raise RuntimeError(
            f"{scenario}: subgroup support mismatch. Expected "
            f"{EXPECTED_SUPPORT[scenario]}, observed {observed_support}"
        )

    matrix_array = confusion_matrix(
        group_oof["y_true"],
        group_oof["y_pred"],
        labels=classes,
    )
    if int(matrix_array.sum()) != EXPECTED_GROUPS:
        raise RuntimeError("Confusion matrix must sum to 225")
    matrix_counts = pd.DataFrame(matrix_array, index=classes, columns=classes)
    matrix_counts.index.name = "true_class"

    row_totals = matrix_array.sum(axis=1, keepdims=True)
    normalized_array = np.divide(
        matrix_array,
        row_totals,
        out=np.zeros_like(matrix_array, dtype=float),
        where=row_totals != 0,
    )
    matrix_normalized = pd.DataFrame(
        normalized_array, index=classes, columns=classes
    )
    matrix_normalized.index.name = "true_class"

    fold_distribution = (
        validation_assignment.pivot_table(
            index="fold",
            columns="class",
            values="subgroup_id",
            aggfunc="nunique",
            fill_value=0,
        )
        .reindex(columns=classes, fill_value=0)
        .reset_index()
    )
    fold_distribution["total_subgroups"] = fold_distribution[classes].sum(axis=1)
    fold_distribution["validation_images"] = (
        fold_distribution["total_subgroups"] * 72
    )

    overall_rows = [
        {
            "scenario": scenario,
            "feature": FEATURE,
            "model": model_name,
            "evaluation_level": "image",
            "n_predictions": EXPECTED_IMAGES,
            **pooled_image,
        },
        {
            "scenario": scenario,
            "feature": FEATURE,
            "model": model_name,
            "evaluation_level": "tissue_subgroup",
            "n_predictions": EXPECTED_GROUPS,
            **pooled_group,
        },
    ]
    overall = pd.DataFrame(overall_rows)

    balanced_summary_rows = []
    for level, column in [
        ("image", "image_balanced_accuracy"),
        ("tissue_subgroup", "subgroup_balanced_accuracy"),
    ]:
        balanced_summary_rows.append(
            {
                "scenario": scenario,
                "evaluation_level": level,
                "five_fold_mean": fold_metrics[column].mean(),
                "five_fold_sample_sd": fold_metrics[column].std(ddof=1),
                "minimum_fold": fold_metrics[column].min(),
                "maximum_fold": fold_metrics[column].max(),
            }
        )
    balanced_summary = pd.DataFrame(balanced_summary_rows)

    comparison = compare_metrics(
        source_summary,
        source_folds,
        fold_metrics,
        pooled_image,
        pooled_group,
        classwise,
        args.tolerance,
    )
    mismatches = comparison[comparison["match"].eq(False)]

    validation_assignment.to_csv(
        scenario_out / "AUTHORITATIVE_VALIDATION_FOLD_ASSIGNMENT_225.csv",
        index=False,
    )
    split_assignment.to_csv(
        scenario_out / "FULL_TRAIN_VALIDATION_FOLD_ASSIGNMENT.csv",
        index=False,
    )
    leakage_audit.to_csv(
        scenario_out / "TRAIN_VALIDATION_LEAKAGE_AUDIT.csv",
        index=False,
    )
    image_oof.to_csv(
        scenario_out / "OOF_IMAGE_PREDICTIONS_16200.csv",
        index=False,
    )
    group_oof.to_csv(
        scenario_out / "OOF_SUBGROUP_PREDICTIONS_225.csv",
        index=False,
    )
    fold_metrics.to_csv(
        scenario_out / "FOLD_IMAGE_AND_SUBGROUP_METRICS.csv",
        index=False,
    )
    fold_distribution.to_csv(
        scenario_out / "FOLD_CLASS_DISTRIBUTION.csv",
        index=False,
    )
    overall.to_csv(
        scenario_out / "POOLED_IMAGE_AND_SUBGROUP_OVERALL_METRICS.csv",
        index=False,
    )
    balanced_summary.to_csv(
        scenario_out / "BALANCED_ACCURACY_FIVE_FOLD_SUMMARY.csv",
        index=False,
    )
    classwise.to_csv(
        scenario_out / "POOLED_SUBGROUP_PER_CLASS_METRICS.csv",
        index=False,
    )
    matrix_counts.to_csv(
        scenario_out / "POOLED_SUBGROUP_CONFUSION_COUNTS.csv"
    )
    matrix_normalized.to_csv(
        scenario_out / "POOLED_SUBGROUP_CONFUSION_ROW_NORMALIZED.csv"
    )
    comparison.to_csv(
        scenario_out / "REGENERATION_VS_ORIGINAL_AUDIT.csv",
        index=False,
    )

    report = scenario_out / "SELECTED_MODEL_FINAL_MEDICAL_REPORT.txt"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("SELECTED CLASSICAL-ML MODEL — CLEAN OOF MEDICAL REPORT\n")
        handle.write("=" * 120 + "\n\n")
        handle.write(
            f"Scenario: {scenario}\nFeature: {FEATURE}\nClassifier: {model_name}\n"
            f"Run settings: {settings_path}\nSeed: {seed}\nFolds: {folds}\n"
            f"n_estimators: {n_estimators}\nmax_iter: {max_iter}\n"
            f"scale_dim_limit: {scale_dim_limit}\n"
            f"Feature cache: {cache_source}\n"
            f"Observed subgroup aggregation: {sorted(aggregation_methods)}\n\n"
        )
        handle.write("POOLED IMAGE AND SUBGROUP METRICS\n" + "-" * 120 + "\n")
        handle.write(overall.to_string(index=False))
        handle.write("\n\nFIVE-FOLD BALANCED ACCURACY\n" + "-" * 120 + "\n")
        handle.write(balanced_summary.to_string(index=False))
        handle.write("\n\nPER-CLASS SUBGROUP METRICS\n" + "-" * 120 + "\n")
        handle.write(classwise.to_string(index=False))
        handle.write("\n\nCONFUSION MATRIX — COUNTS\n" + "-" * 120 + "\n")
        handle.write(matrix_counts.to_string())
        handle.write("\n\nCONFUSION MATRIX — ROW NORMALIZED\n" + "-" * 120 + "\n")
        handle.write(matrix_normalized.to_string())
        handle.write("\n\nFOLD CLASS DISTRIBUTION\n" + "-" * 120 + "\n")
        handle.write(fold_distribution.to_string(index=False))
        handle.write("\n\nFOLD IMAGE AND SUBGROUP METRICS\n" + "-" * 120 + "\n")
        handle.write(fold_metrics.to_string(index=False))
        handle.write("\n\nLEAKAGE AUDIT\n" + "-" * 120 + "\n")
        handle.write(leakage_audit.to_string(index=False))
        handle.write("\n\nREGENERATION MATCH AUDIT\n" + "-" * 120 + "\n")
        handle.write(comparison.to_string(index=False))
        handle.write("\n")

    run_record = {
        "status": "complete" if mismatches.empty else "metric_mismatch",
        "generated_at": datetime.now().isoformat(),
        "scenario": scenario,
        "feature": FEATURE,
        "model": model_name,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "training_script": str(training_script),
        "training_script_sha256": sha256(training_script),
        "run_settings": settings_path,
        "seed": seed,
        "folds": folds,
        "n_estimators": n_estimators,
        "max_iter": max_iter,
        "scale_dim_limit": scale_dim_limit,
        "oof_image_predictions": len(image_oof),
        "oof_subgroup_predictions": len(group_oof),
        "confusion_matrix_total": int(matrix_array.sum()),
        "all_fold_leakage_checks_passed": bool(leakage_audit["pass"].all()),
        "aggregation_methods": sorted(aggregation_methods),
        "sample_weight_used_folds": int(
            fold_metrics["sample_weight_applied"].sum()
        ),
        "sample_weight_notes": sorted(
            fold_metrics["sample_weight_note"].unique().tolist()
        ),
        "metric_mismatches": len(mismatches),
    }
    (scenario_out / "RUN_COMPLETE.json").write_text(
        json.dumps(run_record, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(f"{scenario.upper()} REGENERATION FINISHED")
    print("=" * 100)
    print(json.dumps(run_record, indent=2))
    print()
    print(overall.to_string(index=False))
    print()
    print(classwise.to_string(index=False))
    print()

    if not mismatches.empty:
        print("[FAIL] Regenerated metrics differ from the original result:")
        print(mismatches.to_string(index=False))
        raise SystemExit(2)

    print("[PASS] Exact regenerated metrics match the selected original result.")


def finalize(args) -> None:
    root = args.root
    output_root = root / "selected_best_oof_regeneration_v3"

    scenario_payloads = {}
    for scenario in SELECTED:
        scenario_root = output_root / scenario
        complete_path = scenario_root / "RUN_COMPLETE.json"
        require_file(complete_path)
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "complete":
            raise RuntimeError(
                f"{scenario} is not complete: {complete.get('status')}"
            )

        files = {
            "assignment": "AUTHORITATIVE_VALIDATION_FOLD_ASSIGNMENT_225.csv",
            "split_assignment": "FULL_TRAIN_VALIDATION_FOLD_ASSIGNMENT.csv",
            "leakage": "TRAIN_VALIDATION_LEAKAGE_AUDIT.csv",
            "image_oof": "OOF_IMAGE_PREDICTIONS_16200.csv",
            "group_oof": "OOF_SUBGROUP_PREDICTIONS_225.csv",
            "fold_metrics": "FOLD_IMAGE_AND_SUBGROUP_METRICS.csv",
            "distribution": "FOLD_CLASS_DISTRIBUTION.csv",
            "overall": "POOLED_IMAGE_AND_SUBGROUP_OVERALL_METRICS.csv",
            "balanced": "BALANCED_ACCURACY_FIVE_FOLD_SUMMARY.csv",
            "classwise": "POOLED_SUBGROUP_PER_CLASS_METRICS.csv",
            "audit": "REGENERATION_VS_ORIGINAL_AUDIT.csv",
        }
        scenario_payloads[scenario] = {}
        for key, filename in files.items():
            path = scenario_root / filename
            require_file(path)
            scenario_payloads[scenario][key] = pd.read_csv(path)

    combined_names = {
        "assignment": "AUTHORITATIVE_VALIDATION_FOLD_ASSIGNMENTS_BOTH_SCENARIOS.csv",
        "split_assignment": "FULL_TRAIN_VALIDATION_FOLD_ASSIGNMENTS_BOTH_SCENARIOS.csv",
        "leakage": "TRAIN_VALIDATION_LEAKAGE_AUDIT_BOTH_SCENARIOS.csv",
        "group_oof": "OOF_SUBGROUP_PREDICTIONS_BOTH_SCENARIOS_450.csv",
        "fold_metrics": "FOLD_IMAGE_AND_SUBGROUP_METRICS_BOTH_SCENARIOS.csv",
        "distribution": "FOLD_CLASS_DISTRIBUTION_BOTH_SCENARIOS.csv",
        "overall": "POOLED_IMAGE_AND_SUBGROUP_OVERALL_METRICS_BOTH_SCENARIOS.csv",
        "balanced": "BALANCED_ACCURACY_FIVE_FOLD_SUMMARY_BOTH_SCENARIOS.csv",
        "classwise": "POOLED_SUBGROUP_PER_CLASS_METRICS_BOTH_SCENARIOS.csv",
        "audit": "REGENERATION_VS_ORIGINAL_AUDIT_BOTH_SCENARIOS.csv",
    }

    combined = {}
    for key, filename in combined_names.items():
        frame = pd.concat(
            [scenario_payloads[scenario][key] for scenario in SELECTED],
            ignore_index=True,
            sort=False,
        )
        frame.to_csv(output_root / filename, index=False)
        combined[key] = frame

    left = scenario_payloads["scenario1"]["assignment"][
        ["subgroup_id", "class", "fold"]
    ].rename(
        columns={
            "class": "scenario1_class",
            "fold": "scenario1_fold",
        }
    )
    right = scenario_payloads["scenario2"]["assignment"][
        ["subgroup_id", "class", "fold"]
    ].rename(
        columns={
            "class": "scenario2_class",
            "fold": "scenario2_fold",
        }
    )
    comparison = left.merge(
        right,
        on="subgroup_id",
        how="outer",
        validate="one_to_one",
    )
    comparison["same_fold"] = (
        comparison["scenario1_fold"] == comparison["scenario2_fold"]
    )
    comparison.to_csv(
        output_root / "SCENARIO1_VS_SCENARIO2_FOLD_COMPARISON.csv",
        index=False,
    )

    same_count = int(comparison["same_fold"].sum())
    final_report = output_root / "FINAL_SELECTED_MODELS_MEDICAL_RESULTS.txt"
    with final_report.open("w", encoding="utf-8") as handle:
        handle.write("FINAL SELECTED CLASSICAL-ML MEDICAL RESULTS\n")
        handle.write("=" * 130 + "\n\n")
        handle.write(
            "VALIDATION\n"
            "----------\n"
            "- Scenario 1 pooled image predictions: 16,200\n"
            "- Scenario 1 pooled subgroup predictions: 225\n"
            "- Scenario 2 pooled image predictions: 16,200\n"
            "- Scenario 2 pooled subgroup predictions: 225\n"
            "- Each pooled confusion matrix sums to 225\n"
            "- Every subgroup is held out exactly once per scenario\n"
            "- Train/validation subgroup intersection is empty in every fold\n"
            "- Regenerated selected-model metrics match the original result\n\n"
        )
        handle.write("POOLED IMAGE AND SUBGROUP METRICS\n" + "-" * 130 + "\n")
        handle.write(combined["overall"].to_string(index=False))
        handle.write("\n\nFIVE-FOLD BALANCED ACCURACY\n" + "-" * 130 + "\n")
        handle.write(combined["balanced"].to_string(index=False))
        handle.write("\n\nPER-CLASS SUBGROUP METRICS\n" + "-" * 130 + "\n")
        handle.write(combined["classwise"].to_string(index=False))
        handle.write("\n\nFOLD CLASS DISTRIBUTIONS\n" + "-" * 130 + "\n")
        handle.write(combined["distribution"].to_string(index=False))
        handle.write("\n\nFOLD IMAGE AND SUBGROUP METRICS\n" + "-" * 130 + "\n")
        handle.write(combined["fold_metrics"].to_string(index=False))
        handle.write("\n\nFOLD ASSIGNMENT CONCLUSION\n" + "-" * 130 + "\n")
        handle.write(
            f"Scenario 1 and Scenario 2 same-fold assignments: {same_count}/225.\n"
        )
        if same_count == 225:
            handle.write(
                "The two scenarios used the same subgroup-to-fold mapping.\n"
            )
        else:
            handle.write(
                "The two scenarios did not use an identical subgroup-to-fold mapping. "
                "Do not claim cross-scenario fold identity.\n"
            )
        handle.write(
            "Classical-ML versus CNN fold identity remains unverified until a "
            "CNN subgroup-to-fold file is compared.\n"
        )

    complete = {
        "status": "complete",
        "generated_at": datetime.now().isoformat(),
        "scenario1_oof_images": 16_200,
        "scenario1_oof_subgroups": 225,
        "scenario2_oof_images": 16_200,
        "scenario2_oof_subgroups": 225,
        "confusion_matrix_total_each_scenario": 225,
        "same_cross_scenario_fold_assignments": same_count,
        "different_cross_scenario_fold_assignments": 225 - same_count,
        "classical_ml_vs_cnn_fold_identity": "not_verified",
    }
    (output_root / "FINAL_RUN_COMPLETE.json").write_text(
        json.dumps(complete, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("FINAL SELECTED-MODEL MEDICAL PACKAGE COMPLETE")
    print("=" * 100)
    print(json.dumps(complete, indent=2))
    print(f"\nReport: {final_report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--scenario",
        choices=list(SELECTED),
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
    )
    args = parser.parse_args()

    if args.finalize:
        finalize(args)
        return
    if args.scenario is None:
        parser.error("Use --scenario scenario1, --scenario scenario2, or --finalize")
    run_scenario(args, args.scenario)


if __name__ == "__main__":
    main()
