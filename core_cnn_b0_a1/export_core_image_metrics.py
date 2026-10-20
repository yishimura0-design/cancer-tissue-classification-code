#!/usr/bin/env python3
"""Aggregate held-out Core and Single-Image metrics for cancer B0/A1.

Read-only: uses saved predictions, never retrains or changes the splits.
Outputs contain aggregate metrics only (no core IDs, image paths, or predictions).
Incomplete configurations are marked incomplete, never given pooled scores.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)

from publish_ccast_summary import (
    CNN_MODELS, FOLDS, SCENARIOS, assert_variant_roles, classical_fold,
    cnn_fold, read_roles, source_experiment,
)

MODELS = (*CNN_MODELS, "inner_selected_classical")
LEVELS = ("core", "single_image")


def describe(y: pd.Series, predicted: pd.Series, classes: list[str]) -> tuple[dict, list[dict]]:
    y, predicted = y.astype(str), predicted.astype(str)
    if not set(y).issubset(classes) or not set(predicted).issubset(classes):
        raise ValueError("Unknown class in held-out predictions")
    matrix = confusion_matrix(y, predicted, labels=classes)
    support = matrix.sum(axis=1)
    if any(s == 0 for s in support):
        raise ValueError("Cannot calculate pooled balanced accuracy with missing classes")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, labels=classes, zero_division=0
    )
    summary = {
        "n_predictions": int(len(y)),
        "accuracy": round(float(accuracy_score(y, predicted)), 6),
        "balanced_accuracy": round(float(recall.mean()), 6),
        "macro_f1": round(float(f1_score(y, predicted, labels=classes, average="macro", zero_division=0)), 6),
        "confusion_matrix": matrix.tolist(),
    }
    rows = [
        {
            "class": name,
            "support": int(support[i]),
            "correct": int(matrix[i, i]),
            "predicted": int(matrix[:, i].sum()),
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
        }
        for i, name in enumerate(classes)
    ]
    return summary, rows


@lru_cache(maxsize=4)
def read_image_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validated_fold(root: Path, roles: pd.DataFrame, exp: str, scenario: str,
                   model: str, fold: int):
    source = source_experiment(exp, fold)
    if model == "inner_selected_classical":
        core = classical_fold(root, source, scenario, fold, roles)
        image_file = root / source / "classical" / "selected_image_oof.csv.gz"
    else:
        core = cnn_fold(root, source, scenario, model, fold, roles)
        image_file = (root / source / "cnn" / scenario / model /
                      f"fold_{fold}" / "test_image_predictions.csv.gz")
    if core is None or not image_file.is_file() or image_file.stat().st_size == 0:
        return None

    image = read_image_csv(image_file)
    if model == "inner_selected_classical":
        image = image[(image["scenario"] == scenario) & (image["fold"] == fold)].copy()
        image = image.rename(columns={"y_true": "target_label", "y_pred": "prediction_label"})
    required = {"core_id", "fov_id", "path", "target_label", "prediction_label"}
    if not required.issubset(image.columns):
        raise ValueError(f"{exp}/{scenario}/{model}/fold{fold}: image prediction columns missing")
    if "fold" in image and not (image["fold"] == fold).all():
        raise ValueError("Image fold identity mismatch")
    if "experiment_id" in image and not (image["experiment_id"] == source).all():
        raise ValueError("Image experiment identity mismatch")
    if "scenario" in image and not (image["scenario"] == scenario).all():
        raise ValueError("Image scenario identity mismatch")
    if model != "inner_selected_classical" and "model" in image and not (image["model"] == model).all():
        raise ValueError("Image model identity mismatch")
    if len(image) != 3240 or image.path.isna().any() or image.path.duplicated().any():
        raise ValueError(f"{exp}/{scenario}/{model}/fold{fold}: image coverage is not 3,240 unique paths")
    if image[["core_id", "fov_id", "target_label", "prediction_label"]].isna().any().any():
        raise ValueError("Missing image identity or labels")

    test_roles = roles[(roles.outer_fold == fold) & (roles.role == "test")]
    expected = test_roles.set_index("core_id")[
        "scenario1_label" if scenario == "scenario1" else "scenario2_label"
    ]
    if len(expected) != 9 or set(image.core_id) != set(expected.index):
        raise ValueError("Image predictions do not cover the exact held-out cores")
    if not (image.core_id.value_counts().reindex(expected.index) == 360).all():
        raise ValueError("Each held-out Core must have exactly 360 image predictions")
    if not (image.groupby(["core_id", "fov_id"]).size() == 72).all():
        raise ValueError("Each held-out FOV must have exactly 72 image predictions")
    if not (image.groupby("core_id").fov_id.nunique() == 5).all():
        raise ValueError("Each held-out Core must have five distinct FOVs")
    if not (image.target_label == image.core_id.map(expected)).all():
        raise ValueError("Image ground truth differs from frozen Core role manifest")
    if not set(image.prediction_label).issubset(SCENARIOS[scenario]):
        raise ValueError("Unknown predicted class in images")
    return core[["core_id", "target_label", "prediction_label"]], image[
        ["core_id", "path", "target_label", "prediction_label"]
    ]


def summarize(root: Path, baseline_roles: Path, variant_roles: Path, out: Path,
              require_complete: bool = False) -> dict:
    b0 = read_roles(baseline_roles)
    a1 = read_roles(variant_roles)
    assert_variant_roles(b0, a1)
    rows, per_class_rows, records = [], [], []

    for exp, roles in (("B0", b0), ("A1", a1)):
        for scenario, classes in SCENARIOS.items():
            for model in MODELS:
                cores, images, missing = [], [], []
                for fold in FOLDS:
                    result = validated_fold(root, roles, exp, scenario, model, fold)
                    if result is None:
                        missing.append(fold)
                    else:
                        c, im = result
                        cores.append(c)
                        images.append(im)
                base = {"experiment": exp, "scenario": scenario, "model": model,
                        "completed_folds": len(cores), "missing_folds": missing}
                if missing:
                    rows.append({**base, "status": "incomplete",
                                 "core_accuracy": None, "core_balanced_accuracy": None,
                                 "single_image_accuracy": None, "single_image_balanced_accuracy": None})
                    records.append({**base, "status": "incomplete"})
                    continue

                core = pd.concat(cores, ignore_index=True)
                image = pd.concat(images, ignore_index=True)
                if len(core) != 45 or core.core_id.nunique() != 45:
                    raise ValueError("Invalid pooled Core coverage")
                if len(image) != 16200 or image.path.duplicated().any():
                    raise ValueError("Invalid pooled Single Image coverage")
                if not (image.core_id.value_counts() == 360).all() or image.core_id.nunique() != 45:
                    raise ValueError("Invalid Core membership in pooled Single Image predictions")

                core_metrics, core_classes = describe(core.target_label, core.prediction_label, classes)
                image_metrics, image_classes = describe(image.target_label, image.prediction_label, classes)
                rows.append({
                    **base, "status": "complete",
                    "core_accuracy": core_metrics["accuracy"],
                    "core_balanced_accuracy": core_metrics["balanced_accuracy"],
                    "single_image_accuracy": image_metrics["accuracy"],
                    "single_image_balanced_accuracy": image_metrics["balanced_accuracy"],
                })
                records.append({**base, "status": "complete",
                                "core": core_metrics, "single_image": image_metrics})
                for level, metrics in (("core", core_classes), ("single_image", image_classes)):
                    per_class_rows.extend({**base, "level": level, **rec} for rec in metrics)

    if require_complete and any(row["status"] != "complete" for row in rows):
        raise ValueError("Cannot publish a complete comparison: some predictions are missing")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "core_image_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "per_class_metrics.csv", index=False)
    report = {
        "description": "Held-out 5-fold aggregate metrics; one prediction per Core/Image; no patient-level independence claim",
        "balanced_accuracy_definition": "Pooled mean of per-class recall across all classes, not an average of fold scores",
        "core_unit": "45 physical tissue cores per complete configuration",
        "single_image_unit": "16,200 correlated angular images per complete configuration",
        "a1_note": "Exploratory fold-swap sensitivity; reuses B0 folds 1/3/5, not external validation",
        "models": records,
    }
    (out / "core_image_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--baseline-roles", type=Path, default=Path("splits/inner_outer_roles.csv"))
    parser.add_argument("--variant-roles", type=Path, default=Path("splits_variant_a1/inner_outer_roles.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = summarize(args.results_root, args.baseline_roles, args.variant_roles, args.out, args.require_complete)
    for rec in report["models"]:
        if rec["status"] == "complete":
            c, i = rec["core"], rec["single_image"]
            print(f"{rec['experiment']} {rec['scenario']} {rec['model']}: "
                  f"Core acc={c['accuracy']:.4%}, Core balanced={c['balanced_accuracy']:.4%}; "
                  f"Image acc={i['accuracy']:.4%}, Image balanced={i['balanced_accuracy']:.4%}")
        else:
            print(f"{rec['experiment']} {rec['scenario']} {rec['model']}: "
                  f"INCOMPLETE, missing folds={rec['missing_folds']}")
    print("Aggregate-only output:", args.out)


if __name__ == "__main__":
    main()
