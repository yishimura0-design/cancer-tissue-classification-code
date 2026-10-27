#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: professional search-space/model selection using 5-fold subgroup CV.

Purpose:
  - compare all feature search spaces and all candidate classifiers
  - use group_key/subgroup as split unit, not individual images
  - keep all angle variants from the same subgroup in the same fold
  - use subgroup-balanced sample weights during training
  - select best setting by CV subgroup-level Macro F1

No final 30% test set is used here. This stage is for setting selection only.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ml_eval_lib import (
    CLASS_ORDER,
    class_counts,
    clean_value,
    cv_splits_group,
    flatten_mean_std,
    maybe_fit_with_weight,
    metrics_from_pred,
    predict_scores_multiclass,
    subgroup_balanced_weights,
    subgroup_metrics_from_scores,
    write_csv,
)
from models_lib import build_model_pipeline, parse_models, should_skip_model

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def fit_multiclass(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str, seed: int, n_estimators: int, max_iter: int, scale_dim_limit: int, use_group_weights: bool) -> Any:
    model, _ = build_model_pipeline(model_name, seed=seed, n_estimators=n_estimators, max_iter=max_iter, dim=X.shape[1], scale_dim_limit=scale_dim_limit)
    sample_weight = subgroup_balanced_weights(groups) if use_group_weights else None
    maybe_fit_with_weight(model, X, y, sample_weight)
    return model


def evaluate_cv_for_setting(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_name: str, seed: int, n_splits: int, n_estimators: int, max_iter: int, scale_dim_limit: int, use_group_weights: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fold_rows: List[Dict[str, Any]] = []
    for fold, (tr, va) in enumerate(cv_splits_group(y, groups, n_splits, seed), start=1):
        t0 = time.time()
        model = fit_multiclass(
            X=X[tr],
            y=y[tr],
            groups=groups[tr],
            model_name=model_name,
            seed=seed + fold,
            n_estimators=n_estimators,
            max_iter=max_iter,
            scale_dim_limit=scale_dim_limit,
            use_group_weights=use_group_weights,
        )
        y_pred, scores = predict_scores_multiclass(model, X[va])
        image_m = metrics_from_pred(y[va], y_pred, prefix="image")
        group_m = subgroup_metrics_from_scores(y[va], groups[va], scores, prefix="group")
        row = {
            "fold": fold,
            "train_images": int(len(tr)),
            "val_images": int(len(va)),
            "train_groups": int(len(np.unique(groups[tr]))),
            "val_groups": int(len(np.unique(groups[va]))),
            "seconds": round(time.time() - t0, 2),
            **image_m,
            **group_m,
        }
        fold_rows.append({k: clean_value(v) for k, v in row.items()})
    summary = flatten_mean_std([{k: v for k, v in r.items() if k not in {"fold"}} for r in fold_rows], prefix="cv")
    return summary, fold_rows


def sort_key(row: Dict[str, Any], best_metric: str) -> Tuple[float, float, float]:
    def val(k: str) -> float:
        try:
            return float(row.get(k, -1.0))
        except Exception:
            return -1.0
    return (val(best_metric), val("cv_group_accuracy_mean"), val("cv_image_macro_f1_mean"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", default="results_angle_cpu/stage1_5fold_search")
    ap.add_argument("--models", default="all")
    ap.add_argument("--only-spaces", default="")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-iter", type=int, default=8000)
    ap.add_argument("--scale-dim-limit", type=int, default=400_000)
    ap.add_argument("--use-group-weights", action="store_true", default=True)
    ap.add_argument("--no-group-weights", action="store_false", dest="use_group_weights")
    ap.add_argument("--best-metric", default="cv_group_macro_f1_mean")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    models = parse_models(args.models)
    wanted_spaces = {x.strip() for x in args.only_spaces.split(",") if x.strip()}
    npz_files = sorted(Path(args.features_dir).glob("*.npz"))
    if wanted_spaces:
        npz_files = [p for p in npz_files if p.stem in wanted_spaces]

    print("===== STAGE 1: 5-fold subgroup CV search =====")
    print("Feature caches:", [p.name for p in npz_files])
    print("Models:", models)
    print("Best metric:", args.best_metric)
    print("Group-balanced weights:", args.use_group_weights)

    all_rows: List[Dict[str, Any]] = []
    fold_rows_all: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int64, copy=False)
        groups = data["group_keys"].astype(str) if "group_keys" in data.files else data["image_ids"].astype(str)
        spec_json = str(data["spec_json"][0]) if "spec_json" in data.files else "{}"
        search_space = npz_path.stem
        counts = class_counts(y)
        n_groups = int(len(np.unique(groups)))
        group_counts = {cls: int(len(np.unique(groups[y == i]))) for i, cls in enumerate(CLASS_ORDER)}

        print(f"\n===== {search_space}: X={X.shape}, images={len(y)}, groups={n_groups}, counts={counts}, group_counts={group_counts} =====")

        for model_name in models:
            base: Dict[str, Any] = {
                "search_space": search_space,
                "model": model_name,
                "method": "multiclass_stage1_search",
                "n_samples": int(X.shape[0]),
                "dim": int(X.shape[1]),
                "n_groups": n_groups,
                "n_splits": args.n_splits,
                "seed": args.seed,
                "group_weighted_training": int(args.use_group_weights),
                "feature_spec_json": spec_json,
                **{f"count_{k}": v for k, v in counts.items()},
                **{f"group_count_{k}": v for k, v in group_counts.items()},
            }
            skip = should_skip_model(model_name, n_train=int(X.shape[0] * 0.8), dim=X.shape[1])
            if skip is not None:
                row = {**base, "status": "skipped", "skip_reason": skip}
                all_rows.append(row)
                print(f"SKIP {search_space} / {model_name}: {skip}")
                continue
            print(f"Training CV {search_space} / {model_name} ...", flush=True)
            try:
                t0 = time.time()
                cv_summary, fold_rows = evaluate_cv_for_setting(
                    X=X,
                    y=y,
                    groups=groups,
                    model_name=model_name,
                    seed=args.seed,
                    n_splits=args.n_splits,
                    n_estimators=args.n_estimators,
                    max_iter=args.max_iter,
                    scale_dim_limit=args.scale_dim_limit,
                    use_group_weights=args.use_group_weights,
                )
                row = {**base, "status": "ok", **cv_summary, "total_seconds": round(time.time() - t0, 2)}
                row = {k: clean_value(v) for k, v in row.items()}
                all_rows.append(row)
                for fr in fold_rows:
                    fold_rows_all.append({"search_space": search_space, "model": model_name, **fr})
                print(
                    f"DONE {search_space} / {model_name}: "
                    f"cv_group_macroF1={row.get('cv_group_macro_f1_mean', '')} ± {row.get('cv_group_macro_f1_std', '')}, "
                    f"cv_image_macroF1={row.get('cv_image_macro_f1_mean', '')}"
                )
            except Exception as e:
                tb = traceback.format_exc()
                print(f"ERROR {search_space} / {model_name}: {e}")
                error_rows.append({**base, "error": repr(e), "traceback": tb})
                all_rows.append({**base, "status": "error", "error": repr(e)})

            write_csv(out / "all_results_partial.csv", all_rows)
            write_csv(out / "fold_results_partial.csv", fold_rows_all)
            write_csv(out / "errors_partial.csv", error_rows)

    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    ranked = sorted(ok_rows, key=lambda r: sort_key(r, args.best_metric), reverse=True)
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    write_csv(out / "all_results.csv", all_rows)
    write_csv(out / "fold_results.csv", fold_rows_all)
    write_csv(out / "errors.csv", error_rows)
    write_csv(out / "ranked_results.csv", ranked)

    if ranked:
        best = ranked[0]
        best_setting = {
            "stage": "stage1_5fold_subgroup_cv_selection",
            "best_metric": args.best_metric,
            "best_metric_value": best.get(args.best_metric, ""),
            "search_space": best["search_space"],
            "model": best["model"],
            "method_for_selection": "multiclass",
            "class_order": CLASS_ORDER,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_iter": args.max_iter,
            "scale_dim_limit": args.scale_dim_limit,
            "use_group_weights": args.use_group_weights,
            "best_row": best,
            "note": "Use this setting in Stage 2. Stage 1 uses CV only and does not train the final 70%-split test model.",
        }
        with open(out / "stage1_best_setting.json", "w", encoding="utf-8") as f:
            json.dump(best_setting, f, indent=2)

    print("\n===== TOP 10 STAGE 1 SETTINGS =====")
    for r in ranked[:10]:
        print(
            f"#{r['rank']} {r['search_space']} / {r['model']} | "
            f"CV group MacroF1={r.get('cv_group_macro_f1_mean', '')} ± {r.get('cv_group_macro_f1_std', '')} | "
            f"CV image MacroF1={r.get('cv_image_macro_f1_mean', '')}"
        )
    print("\nSaved:", out)
    print("Best setting:", out / "stage1_best_setting.json")


if __name__ == "__main__":
    main()
