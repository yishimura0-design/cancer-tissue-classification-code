#!/usr/bin/env python3
"""Generate a small, de-identified research report for GitHub publication.

Read ONLY stored CCast results; do not train, change splits, submit jobs, or
publish filenames, core identifiers, images, detailed predictions, or checkpoints.
Incomplete runs are explicitly PARTIAL, never presented as pooled CV results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

SCENARIOS = {
    "scenario1": ["Normal", "Malignant"],
    "scenario2": ["Stage IB", "Stage IIA", "Stage IIB", "Stage III", "Stage IV", "Normal"],
}
CNN_MODELS = ("resnet18", "efficientnet_b0", "mobilenet_v3_large")
FOLDS = (1, 2, 3, 4, 5)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_roles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"outer_fold", "core_id", "role", "scenario1_label", "scenario2_label"}
    if not required.issubset(df.columns):
        raise ValueError("Invalid roles manifest")
    if set(df.outer_fold) != set(FOLDS):
        raise ValueError("Invalid outer folds")
    for fold in FOLDS:
        part = df[df.outer_fold == fold]
        if len(part) != 45 or part.core_id.nunique() != 45 or part.role.value_counts().to_dict() != {"train":29, "test":9, "val":7}:
            raise ValueError(f"Invalid core-level split in fold {fold}")
    return df


def assert_variant_roles(base: pd.DataFrame, variant: pd.DataFrame) -> None:
    for fold in FOLDS:
        a = base[base.outer_fold == fold].sort_values("core_id").reset_index(drop=True)
        b = variant[variant.outer_fold == fold].sort_values("core_id").reset_index(drop=True)
        if not a.core_id.equals(b.core_id):
            raise ValueError(f"Variant core table changed in fold {fold}")
        if fold in (1, 3, 5) and not a.equals(b):
            raise ValueError(f"Reused fold {fold} changed")
        if set(a.loc[a.role == 'val', 'core_id']) != set(b.loc[b.role == 'val', 'core_id']):
            raise ValueError(f"Variant validation changed in fold {fold}")
    for fold, original, switched in ((2, "C2", "C3"), (4, "C3", "C2")):
        a = base[base.outer_fold == fold].set_index("core_id")
        b = variant[variant.outer_fold == fold].set_index("core_id")
        if not (a.at[original, "role"] == "test" and b.at[original, "role"] == "train" and
                a.at[switched, "role"] == "train" and b.at[switched, "role"] == "test"):
            raise ValueError("Unexpected A1 variant swap")
        diff = (a.role != b.role)
        if set(diff.index[diff]) != {original, switched}:
            raise ValueError("Unexpected other A1 changes")


def source_experiment(experiment: str, fold: int) -> str:
    return "A1" if experiment == "A1" and fold in (2, 4) else "B0"


def verify_predictions(core: pd.DataFrame, roles: pd.DataFrame, scenario: str, fold: int,
                       classes: list[str]) -> pd.DataFrame:
    part = roles[(roles.outer_fold == fold) & (roles.role == "test")]
    if len(core) != 9 or "core_id" not in core or core.core_id.duplicated().any() or set(core.core_id) != set(part.core_id):
        raise ValueError(f"Fold {fold} has incorrect held-out Core coverage")
    required = {"target_label", "prediction_label"}
    if not required.issubset(core.columns):
        raise ValueError("Missing prediction columns")
    y = core.set_index("core_id").loc[part.core_id].target_label
    expected = part["scenario1_label" if scenario == "scenario1" else "scenario2_label"].to_numpy()
    if list(y) != list(expected):
        raise ValueError("Test truth differs from frozen split labels")
    if not set(core.prediction_label).issubset(classes):
        raise ValueError("Unexpected predicted label")
    return core


def per_class(y: pd.Series, pred: pd.Series, classes: list[str]) -> list[dict]:
    rows = []
    for cls in classes:
        n = int((y == cls).sum())
        pred_count = int((pred == cls).sum())
        correct = int(((y == cls) & (pred == cls)).sum())
        rows.append({"class": cls, "support": n, "predicted": pred_count,
                     "recall": round(correct/n, 6) if n else None,
                     "recall_status": "evaluated" if n else "N/A_no_true_test_core_for_class"})
    return rows


def score_for_fold(core: pd.DataFrame, scenario: str, model: str, fold: int,
                   source: str) -> dict:
    classes = SCENARIOS[scenario]
    y = core.target_label.astype(str)
    pred = core.prediction_label.astype(str)
    return {"scenario": scenario, "model": model, "fold": fold, "source_experiment": source,
            "n_test_cores": int(len(core)),
            "core_accuracy": round(float(accuracy_score(y, pred)), 6),
            "core_macro_f1_all_classes": round(float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)), 6),
            "per_class": per_class(y, pred, classes)}


def cnn_fold(root: Path, experiment: str, scenario: str, model: str, fold: int,
             roles: pd.DataFrame) -> pd.DataFrame | None:
    d = root / experiment / "cnn" / scenario / model / f"fold_{fold}"
    files = [d / "test_core_predictions.csv", d / "test_image_predictions.csv.gz", d / "fold_metrics.json", d / "run_contract.json"]
    if not any(p.exists() for p in files):
        return None
    if not all(p.is_file() and p.stat().st_size > 0 for p in files):
        # An in-flight Fold is not a completed result; never publish metrics from it.
        return None
    m = json.loads(files[2].read_text(encoding="utf-8"))
    identity = json.loads(files[3].read_text(encoding="utf-8"))
    if m.get('experiment_id') != experiment or m.get('scenario') != scenario or m.get('model') != model or int(m.get('fold',-1)) != fold:
        raise ValueError("CNN output has mismatched experiment identity")
    if m.get('outer_test_used_for_checkpoint_selection') is not False:
        raise ValueError("CNN output has no no-test-selection guarantee")
    if m.get('run_contract') != identity:
        raise ValueError("CNN input/code/config integrity marker mismatch")
    if m.get('n_outer_test_cores') != 9 or m.get('outer_test_image_count') != 3240:
        raise ValueError("CNN output test coverage inconsistent")
    core = pd.read_csv(files[0])
    return verify_predictions(core, roles, scenario, fold, SCENARIOS[scenario])


def classical_fold(root: Path, experiment: str, scenario: str, fold: int,
                   roles: pd.DataFrame) -> pd.DataFrame | None:
    d = root / experiment / "classical"
    files = [d / "selected_core_oof.csv", d / "selected_image_oof.csv.gz", d / "fold_metrics.json"]
    if not all(f.exists() and f.stat().st_size for f in files):
        return None
    metrics = json.loads(files[2].read_text(encoding="utf-8"))
    record = [x for x in metrics if x.get("scenario") == scenario and int(x.get("fold", -1)) == fold and x.get("experiment_id") == experiment]
    if not record:
        return None
    if len(record) != 1 or record[0].get("n_test_cores") != 9 or record[0].get("n_test_images") != 3240:
        raise ValueError("Classical fold metrics invalid")
    core = pd.read_csv(files[0]); core = core[(core.scenario == scenario) & (core.fold == fold)].copy()
    core = core.rename(columns={"y_true":"target_label", "y_pred":"prediction_label"})
    return verify_predictions(core, roles, scenario, fold, SCENARIOS[scenario])


def aggregate_complete(parts: list[pd.DataFrame], scenario: str) -> dict:
    core = pd.concat(parts, ignore_index=True)
    if len(core) != 45 or core.core_id.nunique() != 45:
        raise ValueError("Cannot publish pooled metric without 45 distinct held-out cores")
    classes = SCENARIOS[scenario]
    y = core.target_label.astype(str); pred = core.prediction_label.astype(str)
    return {"n_core_predictions": 45,
            "core_accuracy": round(float(accuracy_score(y, pred)),6),
            "core_macro_f1_all_classes": round(float(f1_score(y,pred,labels=classes,average="macro",zero_division=0)),6),
            "classes": classes,
            "confusion_matrix": confusion_matrix(y,pred,labels=classes).tolist(),
            "per_class": per_class(y,pred,classes)}


def create_summary(experiment: str, root: Path, baseline_roles: Path,
                   variant_roles: Path, out: Path, require_complete: bool = False) -> dict:
    base = read_roles(baseline_roles)
    roles = read_roles(variant_roles) if experiment == "A1" else base
    if experiment == "A1":
        assert_variant_roles(base, roles)
    fold_rows = []
    models = []
    for scenario in SCENARIOS:
        for model in (*CNN_MODELS, "inner_selected_classical"):
            collected = []
            missing = []
            for fold in FOLDS:
                src = source_experiment(experiment, fold)
                c = (classical_fold(root, src, scenario, fold, roles) if model == "inner_selected_classical"
                     else cnn_fold(root, src, scenario, model, fold, roles))
                if c is None:
                    missing.append(fold)
                    continue
                collected.append(c)
                fold_rows.append(score_for_fold(c, scenario, model, fold, src))
            models.append({"scenario":scenario,"model":model,"completed_folds":len(collected),
                           "missing_folds":missing,"status":"complete" if len(collected)==5 else "partial",
                           "pooled_core_metrics": aggregate_complete(collected, scenario) if len(collected)==5 else None})
    count = sum(x["completed_folds"] for x in models)
    total = len(models) * len(FOLDS)
    report = {
        "schema_version":1, "experiment_id":experiment,
        "status":"complete" if count == total else "partial",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "expected_model_folds":total, "completed_model_folds":count,
        "core_cv_design":"B0 primary; A1 exploratory reuse B0 1/3/5",
        "labels_source":"validated against frozen Core-level roles CSV",
        "baseline_roles_sha256":hash_file(baseline_roles),
        "variant_roles_sha256":hash_file(variant_roles) if experiment == "A1" else None,
        "scope_note":"Core-independent held-out predictions; patient independence unverified. Aggregate metrics only; no patient IDs, core IDs, paths, images or probabilities are published.",
        "models":models,
    }
    if require_complete and count != total:
        raise ValueError(f"Cannot publish COMPLETE: only {count}/{total} model folds present")
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+"\n", encoding="utf-8")
    with (out/"fold_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["scenario","model","fold","source_experiment","n_test_cores","core_accuracy","core_macro_f1_all_classes"])
        w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in fold_rows])
    with (out/"model_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        keys=["scenario","model","status","completed_folds","missing_folds","pooled_core_accuracy","pooled_core_macro_f1_all_classes"]
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in models:
            pooled = r['pooled_core_metrics'] or {}
            w.writerow({"scenario":r['scenario'],"model":r['model'],"status":r['status'],"completed_folds":r['completed_folds'],
                        "missing_folds":','.join(map(str,r['missing_folds'])),
                        "pooled_core_accuracy":pooled.get('core_accuracy'),"pooled_core_macro_f1_all_classes":pooled.get('core_macro_f1_all_classes')})
    lines = [f"# CCast Core-level CV — {experiment}", "", f"**Status: {report['status'].upper()}** — {count}/{total} model × fold outputs validated.",
             "", f"Last generated (UTC): {report['generated_at_utc']}", "",
             "B0 is the primary core-level CV; A1 is an exploratory fold-placement sensitivity experiment.",
             "Only complete five-fold model entries have pooled 45-core scores. Missing folds are not counted as zero.", "",
             "| Scenario | Model | Folds | Pooled 45-Core accuracy | Pooled Macro-F1 |", "|---|---|---:|---:|---:|"]
    for r in models:
        p=r['pooled_core_metrics']
        lines.append(f"| {r['scenario']} | {r['model']} | {r['completed_folds']}/5 | {p['core_accuracy'] if p else 'N/A (partial)'} | {p['core_macro_f1_all_classes'] if p else 'N/A (partial)'} |")
    lines.extend(["", "See `fold_metrics.csv` for validated held-out Fold scores; `summary.json` includes per-class support/recall and confusion matrices for completed models.",
                  "Class recall is N/A when the true Test support is zero; those absent classes are not measured zeros.",
                  "", "**Privacy:** This report contains aggregate metrics only. Full image/core predictions, paths, checkpoints and run logs remain on CCast.",
                  "**Limitation:** The 45 physical cores are the split units; patient identity is not available for independent patient-level evaluation.", ""])
    (out/"README.md").write_text("\n".join(lines),encoding="utf-8")
    return report


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['B0','A1'],required=True)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--baseline-roles',type=Path,required=True)
    p.add_argument('--variant-roles',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    a=p.parse_args()
    report=create_summary(a.experiment,a.results_root,a.baseline_roles,a.variant_roles,a.out,a.require_complete)
    print(json.dumps({k:report[k] for k in ['experiment_id','status','completed_model_folds','expected_model_folds']},indent=2))

if __name__=='__main__':
    main()
