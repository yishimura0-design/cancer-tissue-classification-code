#!/usr/bin/env python3

from __future__ import annotations
import os

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("CANCER_DATA_ROOT", "./private_data"))

OLD_ROOT = (
    ROOT
    / "ml_results_cancer_tissue_grouped_clean_stage1_5fold"
)

NEW_ROOT = (
    ROOT
    / "ml_results_cancer_tissue_missing4_selected6_parallel"
)

FINAL_ROOT = (
    ROOT
    / "ml_results_cancer_tissue_all8_selected6_final"
)

BUILD_ROOT = FINAL_ROOT.with_name(
    FINAL_ROOT.name + ".building"
)


OLD_FEATURES = [
    "resize64_gray",
    "hist_gray64",
    "hist_rgb32",
    "hog128_gray",
]

NEW_FEATURES = [
    "raw_stride8_gray",
    "raw_stride16_gray",
    "resize128_gray",
    "resize64_rgb",
]

FEATURES = OLD_FEATURES + NEW_FEATURES

MODELS = [
    "ExtraTrees",
    "RandomForest",
    "LinearSVC",
    "SGD_LogLoss",
    "DecisionTree",
    "LightGBM",
]

SCENARIOS = [
    "scenario1",
    "scenario2",
]


# This one feature-classifier pair is unsupported in both scenarios.
UNSUPPORTED = {
    ("scenario1", "raw_stride8_gray", "LinearSVC"),
    ("scenario2", "raw_stride8_gray", "LinearSVC"),
}

UNSUPPORTED_REASON = "internal integer-conversion limit"


EXPECTED_PLANNED_CONFIGURATIONS = 96
EXPECTED_COMPLETED_CONFIGURATIONS = 94
EXPECTED_UNSUPPORTED_CONFIGURATIONS = 2
EXPECTED_COMPLETED_FOLD_ROWS = 470
EXPECTED_FOLDS_PER_COMPLETED_CONFIGURATION = 5


MODEL_DISPLAY = {
    "ExtraTrees": "Extra Trees",
    "RandomForest": "Random Forest",
    "LinearSVC": "Linear Support Vector Machine",
    "SGD_LogLoss": (
        "Stochastic Gradient Descent with logistic loss"
    ),
    "DecisionTree": "Decision Tree",
    "LightGBM": "LightGBM",
}

SCENARIO_DISPLAY = {
    "scenario1": "Scenario 1",
    "scenario2": "Scenario 2",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")

    if path.stat().st_size == 0:
        raise RuntimeError(f"Required file is empty: {path}")


def load_old_feature(
    feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_root = OLD_ROOT / feature

    summary_path = feature_root / "results_summary.csv"
    fold_path = feature_root / "fold_results.csv"

    require_file(summary_path)
    require_file(fold_path)

    summary = pd.read_csv(summary_path)
    folds = pd.read_csv(fold_path)

    summary = summary[
        summary["scenario"].isin(SCENARIOS)
        & summary["model"].isin(MODELS)
        & summary["cv_status"]
        .astype(str)
        .str.lower()
        .eq("ok")
    ].copy()

    folds = folds[
        folds["scenario"].isin(SCENARIOS)
        & folds["model"].isin(MODELS)
    ].copy()

    summary["feature"] = feature
    folds["feature"] = feature

    summary["source_type"] = "preserved_existing_result"
    folds["source_type"] = "preserved_existing_result"

    summary["source_dir"] = str(feature_root)
    folds["source_dir"] = str(feature_root)

    expected_summary_rows = (
        len(SCENARIOS) * len(MODELS)
    )

    expected_fold_rows = (
        expected_summary_rows
        * EXPECTED_FOLDS_PER_COMPLETED_CONFIGURATION
    )

    if len(summary) != expected_summary_rows:
        raise RuntimeError(
            f"{feature}: expected {expected_summary_rows} "
            f"existing summary rows, found {len(summary)}"
        )

    if len(folds) != expected_fold_rows:
        raise RuntimeError(
            f"{feature}: expected {expected_fold_rows} "
            f"existing fold rows, found {len(folds)}"
        )

    return summary, folds


def load_new_feature(
    feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    fold_tables: list[pd.DataFrame] = []

    for model in MODELS:
        unsupported_for_feature = any(
            unsupported_feature == feature
            and unsupported_model == model
            for (
                _scenario,
                unsupported_feature,
                unsupported_model,
            ) in UNSUPPORTED
        )

        if unsupported_for_feature:
            print(
                "[SKIP AS DOCUMENTED] "
                f"{feature} / {model}: "
                f"{UNSUPPORTED_REASON}"
            )
            continue

        model_root = NEW_ROOT / feature / model

        summary_path = model_root / "results_summary.csv"
        fold_path = model_root / "fold_results.csv"

        require_file(summary_path)
        require_file(fold_path)

        summary = pd.read_csv(summary_path)
        folds = pd.read_csv(fold_path)

        summary = summary[
            summary["scenario"].isin(SCENARIOS)
            & summary["model"].eq(model)
            & summary["cv_status"]
            .astype(str)
            .str.lower()
            .eq("ok")
        ].copy()

        folds = folds[
            folds["scenario"].isin(SCENARIOS)
            & folds["model"].eq(model)
        ].copy()

        summary["feature"] = feature
        folds["feature"] = feature

        summary["source_type"] = "new_parallel_result"
        folds["source_type"] = "new_parallel_result"

        summary["source_dir"] = str(model_root)
        folds["source_dir"] = str(model_root)

        if len(summary) != 2:
            raise RuntimeError(
                f"{feature}/{model}: expected 2 completed "
                f"scenario rows, found {len(summary)}"
            )

        if len(folds) != 10:
            raise RuntimeError(
                f"{feature}/{model}: expected 10 completed "
                f"fold rows, found {len(folds)}"
            )

        model_fold_counts = (
            folds.groupby(
                ["scenario", "feature", "model"]
            )["fold"]
            .nunique()
        )

        if not model_fold_counts.eq(5).all():
            raise RuntimeError(
                f"{feature}/{model}: one or more scenarios "
                "do not contain five folds"
            )

        summaries.append(summary)
        fold_tables.append(folds)

    if not summaries:
        raise RuntimeError(
            f"No supported results were loaded for {feature}"
        )

    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(fold_tables, ignore_index=True),
    )


def format_mean_std(
    mean_value,
    std_value,
) -> str:
    if pd.isna(mean_value):
        return ""

    if pd.isna(std_value):
        return f"{float(mean_value):.4f}"

    return (
        f"{float(mean_value):.4f} "
        f"± {float(std_value):.4f}"
    )


def add_display_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["scenario_display"] = (
        dataframe["scenario"]
        .map(SCENARIO_DISPLAY)
        .fillna(dataframe["scenario"])
    )

    dataframe["model_display"] = (
        dataframe["model"]
        .map(MODEL_DISPLAY)
        .fillna(dataframe["model"])
    )

    dataframe["feature_display"] = dataframe["feature"]

    return dataframe


def validate_complete_results(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    summary = summary.drop_duplicates(
        subset=["scenario", "feature", "model"],
        keep="last",
    )

    folds = folds.drop_duplicates(
        subset=[
            "scenario",
            "feature",
            "model",
            "fold",
        ],
        keep="last",
    )

    expected_keys = {
        (scenario, feature, model)
        for scenario in SCENARIOS
        for feature in FEATURES
        for model in MODELS
    }

    completed_keys = set(
        summary[
            ["scenario", "feature", "model"]
        ].itertuples(index=False, name=None)
    )

    missing_keys = expected_keys - completed_keys
    unexpected_keys = completed_keys - expected_keys

    if unexpected_keys:
        raise RuntimeError(
            "Unexpected completed configurations:\n"
            + "\n".join(
                str(key)
                for key in sorted(unexpected_keys)
            )
        )

    if missing_keys != UNSUPPORTED:
        raise RuntimeError(
            "Missing configurations do not match the "
            "documented unsupported set.\n\n"
            "Expected unsupported:\n"
            + "\n".join(
                str(key)
                for key in sorted(UNSUPPORTED)
            )
            + "\n\nObserved missing:\n"
            + "\n".join(
                str(key)
                for key in sorted(missing_keys)
            )
        )

    if len(expected_keys) != EXPECTED_PLANNED_CONFIGURATIONS:
        raise RuntimeError(
            "Internal planned-configuration count mismatch"
        )

    if (
        len(completed_keys)
        != EXPECTED_COMPLETED_CONFIGURATIONS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_COMPLETED_CONFIGURATIONS} "
            f"completed configurations, found "
            f"{len(completed_keys)}"
        )

    if (
        len(missing_keys)
        != EXPECTED_UNSUPPORTED_CONFIGURATIONS
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_UNSUPPORTED_CONFIGURATIONS} "
            f"unsupported scenario configurations, found "
            f"{len(missing_keys)}"
        )

    if (
        len(completed_keys) + len(missing_keys)
        != EXPECTED_PLANNED_CONFIGURATIONS
    ):
        raise RuntimeError(
            "Completed plus unsupported configurations "
            "do not account for the complete search space"
        )

    fold_counts = (
        folds.groupby(
            ["scenario", "feature", "model"]
        )["fold"]
        .nunique()
    )

    bad_fold_counts = fold_counts[
        fold_counts
        != EXPECTED_FOLDS_PER_COMPLETED_CONFIGURATION
    ]

    if not bad_fold_counts.empty:
        raise RuntimeError(
            "Some completed configurations do not "
            "contain exactly five folds:\n"
            + bad_fold_counts.to_string()
        )

    fold_keys = set(
        folds[
            ["scenario", "feature", "model"]
        ].drop_duplicates()
        .itertuples(index=False, name=None)
    )

    if fold_keys != completed_keys:
        missing_fold_keys = completed_keys - fold_keys
        unexpected_fold_keys = fold_keys - completed_keys

        raise RuntimeError(
            "Summary and fold configuration keys differ.\n"
            f"Missing fold keys: {sorted(missing_fold_keys)}\n"
            f"Unexpected fold keys: "
            f"{sorted(unexpected_fold_keys)}"
        )

    if len(folds) != EXPECTED_COMPLETED_FOLD_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_COMPLETED_FOLD_ROWS} "
            f"completed fold rows, found {len(folds)}"
        )

    return completed_keys, missing_keys


def create_accounting_table(
    completed_keys: set[tuple[str, str, str]],
) -> pd.DataFrame:
    rows = []

    for scenario in SCENARIOS:
        for feature in FEATURES:
            for model in MODELS:
                key = (scenario, feature, model)

                if key in completed_keys:
                    status = "completed"
                    reason = ""
                    folds_completed = 5
                elif key in UNSUPPORTED:
                    status = "computationally_unsupported"
                    reason = UNSUPPORTED_REASON
                    folds_completed = 0
                else:
                    status = "unexpected_missing"
                    reason = "not accounted for"
                    folds_completed = 0

                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_display": (
                            SCENARIO_DISPLAY[scenario]
                        ),
                        "feature": feature,
                        "model": model,
                        "model_display": (
                            MODEL_DISPLAY[model]
                        ),
                        "status": status,
                        "reason": reason,
                        "folds_completed": folds_completed,
                    }
                )

    accounting = pd.DataFrame(rows)

    if len(accounting) != 96:
        raise RuntimeError(
            "Configuration-accounting table does not "
            "contain 96 rows"
        )

    return accounting


def write_text_tables(
    output_path: Path,
    best: pd.DataFrame,
    top3: pd.DataFrame,
    best_per_model: pd.DataFrame,
    compact: pd.DataFrame,
    accounting: pd.DataFrame,
) -> None:
    display_columns = [
        "scenario_display",
        "feature",
        "model_display",
        "image_macro_f1",
        "subgroup_macro_f1",
        "macro_f1_improvement_pp",
        "image_accuracy",
        "subgroup_accuracy",
        "accuracy_improvement_pp",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "CANCER TISSUE CLASSIFICATION\n"
            "FINAL ALL-EIGHT-FEATURE / "
            "SIX-CLASSIFIER RESULTS\n"
        )
        handle.write("=" * 140 + "\n\n")

        handle.write(
            "Search-space accounting\n"
            "-----------------------\n"
            "Planned scenario-specific configurations: 96\n"
            "Completed scenario-specific configurations: 94\n"
            "Computationally unsupported configurations: 2\n"
            "Completed fold rows: 470\n"
            "Folds per completed configuration: 5\n\n"
        )

        handle.write(
            "Unsupported feature-classifier pair\n"
            "-----------------------------------\n"
            "Feature: raw_stride8_gray\n"
            "Classifier: Linear Support Vector Machine\n"
            "Affected scenarios: Scenario 1 and Scenario 2\n"
            "Reason: internal integer-conversion limit\n\n"
        )

        handle.write(
            "=" * 140
            + "\nBEST RESULT PER SCENARIO\n"
            + "=" * 140
            + "\n"
        )

        handle.write(
            best[display_columns].to_string(index=False)
        )

        handle.write(
            "\n\n"
            + "=" * 140
            + "\nTOP THREE RESULTS PER SCENARIO\n"
            + "=" * 140
            + "\n"
        )

        handle.write(
            top3[display_columns].to_string(index=False)
        )

        handle.write(
            "\n\n"
            + "=" * 140
            + "\nBEST FEATURE FOR EACH CLASSIFIER\n"
            + "=" * 140
            + "\n"
        )

        handle.write(
            best_per_model[
                display_columns
            ].to_string(index=False)
        )

        handle.write(
            "\n\n"
            + "=" * 140
            + "\nALL 94 COMPLETED CONFIGURATIONS\n"
            + "=" * 140
            + "\n"
        )

        for scenario in SCENARIOS:
            scenario_name = SCENARIO_DISPLAY[scenario]

            handle.write(
                f"\n{scenario_name.upper()}\n"
                + "-" * 140
                + "\n"
            )

            scenario_table = compact[
                compact["scenario"] == scenario
            ]

            handle.write(
                scenario_table[
                    display_columns
                ].to_string(index=False)
            )

            handle.write("\n")

        handle.write(
            "\n"
            + "=" * 140
            + "\nCONFIGURATION ACCOUNTING\n"
            + "=" * 140
            + "\n"
        )

        unsupported_rows = accounting[
            accounting["status"]
            != "completed"
        ]

        handle.write(
            unsupported_rows.to_string(index=False)
        )
        handle.write("\n")


def main() -> None:
    print("=" * 100)
    print("LOADING PRESERVED FOUR-FEATURE RESULTS")
    print("=" * 100)

    summary_tables: list[pd.DataFrame] = []
    fold_tables: list[pd.DataFrame] = []

    for feature in OLD_FEATURES:
        summary, folds = load_old_feature(feature)
        summary_tables.append(summary)
        fold_tables.append(folds)

        print(
            f"[PASS] {feature}: "
            f"{len(summary)} configurations, "
            f"{len(folds)} fold rows"
        )

    print()
    print("=" * 100)
    print("LOADING NEW FOUR-FEATURE PARALLEL RESULTS")
    print("=" * 100)

    for feature in NEW_FEATURES:
        summary, folds = load_new_feature(feature)
        summary_tables.append(summary)
        fold_tables.append(folds)

        print(
            f"[PASS] {feature}: "
            f"{len(summary)} completed configurations, "
            f"{len(folds)} fold rows"
        )

    summary = pd.concat(
        summary_tables,
        ignore_index=True,
        sort=False,
    )

    folds = pd.concat(
        fold_tables,
        ignore_index=True,
        sort=False,
    )

    summary = summary.drop_duplicates(
        subset=["scenario", "feature", "model"],
        keep="last",
    )

    folds = folds.drop_duplicates(
        subset=[
            "scenario",
            "feature",
            "model",
            "fold",
        ],
        keep="last",
    )

    completed_keys, missing_keys = (
        validate_complete_results(
            summary,
            folds,
        )
    )

    print()
    print("=" * 100)
    print("ACCOUNTING VALIDATION")
    print("=" * 100)
    print(
        f"[PASS] Planned configurations: "
        f"{EXPECTED_PLANNED_CONFIGURATIONS}"
    )
    print(
        f"[PASS] Completed configurations: "
        f"{len(completed_keys)}"
    )
    print(
        f"[PASS] Unsupported configurations: "
        f"{len(missing_keys)}"
    )
    print(
        f"[PASS] Completed fold rows: "
        f"{len(folds)}"
    )
    print(
        "[PASS] Every completed configuration "
        "contains five folds"
    )

    summary["macro_f1_improvement_pp"] = (
        summary["cv_group_macro_f1_mean"]
        - summary["cv_image_macro_f1_mean"]
    ) * 100

    summary["accuracy_improvement_pp"] = (
        summary["cv_group_acc_mean"]
        - summary["cv_image_acc_mean"]
    ) * 100

    summary["image_macro_f1"] = summary.apply(
        lambda row: format_mean_std(
            row["cv_image_macro_f1_mean"],
            row["cv_image_macro_f1_std"],
        ),
        axis=1,
    )

    summary["subgroup_macro_f1"] = summary.apply(
        lambda row: format_mean_std(
            row["cv_group_macro_f1_mean"],
            row["cv_group_macro_f1_std"],
        ),
        axis=1,
    )

    summary["image_accuracy"] = summary.apply(
        lambda row: format_mean_std(
            row["cv_image_acc_mean"],
            row["cv_image_acc_std"],
        ),
        axis=1,
    )

    summary["subgroup_accuracy"] = summary.apply(
        lambda row: format_mean_std(
            row["cv_group_acc_mean"],
            row["cv_group_acc_std"],
        ),
        axis=1,
    )

    summary = add_display_columns(summary)

    summary = summary.sort_values(
        [
            "scenario",
            "cv_group_macro_f1_mean",
            "cv_group_acc_mean",
        ],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    folds = folds.sort_values(
        [
            "scenario",
            "feature",
            "model",
            "fold",
        ]
    ).reset_index(drop=True)

    best = (
        summary.groupby(
            "scenario",
            group_keys=False,
        )
        .head(1)
        .reset_index(drop=True)
    )

    top3 = (
        summary.groupby(
            "scenario",
            group_keys=False,
        )
        .head(3)
        .reset_index(drop=True)
    )

    best_per_model = (
        summary.sort_values(
            [
                "scenario",
                "model",
                "cv_group_macro_f1_mean",
                "cv_group_acc_mean",
            ],
            ascending=[True, True, False, False],
        )
        .groupby(
            ["scenario", "model"],
            group_keys=False,
        )
        .head(1)
        .reset_index(drop=True)
    )

    accounting = create_accounting_table(
        completed_keys
    )

    unsupported_table = accounting[
        accounting["status"]
        == "computationally_unsupported"
    ].copy()

    numeric_columns = [
        "scenario",
        "scenario_display",
        "feature",
        "model",
        "model_display",
        "n_images",
        "n_subgroups",
        "n_features",
        "cv_image_acc_mean",
        "cv_image_acc_std",
        "cv_image_macro_f1_mean",
        "cv_image_macro_f1_std",
        "cv_image_weighted_f1_mean",
        "cv_image_weighted_f1_std",
        "cv_group_acc_mean",
        "cv_group_acc_std",
        "cv_group_macro_f1_mean",
        "cv_group_macro_f1_std",
        "cv_group_weighted_f1_mean",
        "cv_group_weighted_f1_std",
        "macro_f1_improvement_pp",
        "accuracy_improvement_pp",
        "oof_image_acc",
        "oof_image_macro_f1",
        "oof_image_weighted_f1",
        "oof_group_acc",
        "oof_group_macro_f1",
        "oof_group_weighted_f1",
        "subgroup_aggregation_methods",
        "scaler_used_any_fold",
        "sample_weight_used_folds",
        "sample_weight_notes",
        "source_type",
        "source_dir",
    ]

    numeric_columns = [
        column
        for column in numeric_columns
        if column in summary.columns
    ]

    compact_columns = [
        "scenario",
        "scenario_display",
        "feature",
        "model",
        "model_display",
        "image_macro_f1",
        "subgroup_macro_f1",
        "macro_f1_improvement_pp",
        "image_accuracy",
        "subgroup_accuracy",
        "accuracy_improvement_pp",
        "oof_image_macro_f1",
        "oof_group_macro_f1",
        "oof_image_acc",
        "oof_group_acc",
    ]

    compact_columns = [
        column
        for column in compact_columns
        if column in summary.columns
    ]

    compact = summary[compact_columns].copy()

    if BUILD_ROOT.exists():
        stale_build_backup = BUILD_ROOT.with_name(
            BUILD_ROOT.name
            + ".stale."
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        shutil.move(
            BUILD_ROOT,
            stale_build_backup,
        )

        print(
            "Moved stale build directory to:\n"
            f"{stale_build_backup}"
        )

    BUILD_ROOT.mkdir(parents=True)

    summary[numeric_columns].to_csv(
        BUILD_ROOT
        / "ALL_94_COMPLETED_CONFIGURATION_RESULTS.csv",
        index=False,
    )

    compact.to_csv(
        BUILD_ROOT
        / "ALL_94_RESULTS_COMPACT.csv",
        index=False,
    )

    folds.to_csv(
        BUILD_ROOT
        / "ALL_470_FOLD_RESULTS.csv",
        index=False,
    )

    folds.to_csv(
        BUILD_ROOT
        / "ALL_fold_results.csv",
        index=False,
    )

    best[numeric_columns].to_csv(
        BUILD_ROOT
        / "Table_best_per_scenario.csv",
        index=False,
    )

    top3[numeric_columns].to_csv(
        BUILD_ROOT
        / "Table_top3_per_scenario_compact.csv",
        index=False,
    )

    best_per_model[numeric_columns].to_csv(
        BUILD_ROOT
        / "Table_best_feature_for_each_model.csv",
        index=False,
    )

    accounting.to_csv(
        BUILD_ROOT
        / "CONFIGURATION_ACCOUNTING_96.csv",
        index=False,
    )

    unsupported_table.to_csv(
        BUILD_ROOT
        / "ALL_skipped_or_failed_configurations.csv",
        index=False,
    )

    write_text_tables(
        BUILD_ROOT
        / "ALL8_SELECTED6_FINAL_RESULTS.txt",
        best,
        top3,
        best_per_model,
        compact,
        accounting,
    )

    with (
        BUILD_ROOT
        / "RESULTS_SUMMARY.md"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "# Cancer Tissue Classification Results\n\n"
        )

        handle.write(
            "## Search-space accounting\n\n"
            "- Planned scenario-specific configurations: 96\n"
            "- Completed scenario-specific configurations: 94\n"
            "- Computationally unsupported configurations: 2\n"
            "- Completed fold rows: 470\n"
            "- Folds per completed configuration: 5\n\n"
        )

        handle.write(
            "## Computationally unsupported pair\n\n"
            "- Feature: `raw_stride8_gray`\n"
            "- Classifier: Linear Support Vector Machine\n"
            "- Affected scenarios: Scenario 1 and Scenario 2\n"
            "- Reason: internal integer-conversion limit\n\n"
        )

        handle.write(
            "## Best result per scenario\n\n"
            "```text\n"
        )

        handle.write(
            best[
                [
                    "scenario_display",
                    "feature",
                    "model_display",
                    "image_macro_f1",
                    "subgroup_macro_f1",
                    "macro_f1_improvement_pp",
                    "image_accuracy",
                    "subgroup_accuracy",
                    "accuracy_improvement_pp",
                ]
            ].to_string(index=False)
        )

        handle.write("\n```\n\n")

        handle.write(
            "## Top three results per scenario\n\n"
            "```text\n"
        )

        handle.write(
            top3[
                [
                    "scenario_display",
                    "feature",
                    "model_display",
                    "image_macro_f1",
                    "subgroup_macro_f1",
                    "macro_f1_improvement_pp",
                    "image_accuracy",
                    "subgroup_accuracy",
                    "accuracy_improvement_pp",
                ]
            ].to_string(index=False)
        )

        handle.write("\n```\n")

    run_complete = {
        "status": (
            "complete_with_one_computationally_"
            "unsupported_feature_classifier_pair"
        ),
        "generated_at": datetime.now().isoformat(),
        "planned_scenario_specific_configurations": 96,
        "completed_scenario_specific_configurations": 94,
        "unsupported_scenario_specific_configurations": 2,
        "planned_feature_classifier_pairs": 48,
        "completed_feature_classifier_pairs": 47,
        "unsupported_feature_classifier_pairs": 1,
        "completed_fold_rows": 470,
        "folds_per_completed_configuration": 5,
        "features": FEATURES,
        "models": MODELS,
        "scenarios": SCENARIOS,
        "unsupported_pair": {
            "feature": "raw_stride8_gray",
            "model": "LinearSVC",
            "model_display": (
                MODEL_DISPLAY["LinearSVC"]
            ),
            "affected_scenarios": [
                "scenario1",
                "scenario2",
            ],
            "reason": UNSUPPORTED_REASON,
        },
        "existing_source_root": str(OLD_ROOT),
        "new_parallel_source_root": str(NEW_ROOT),
    }

    with (
        BUILD_ROOT
        / "RUN_COMPLETE.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_complete,
            handle,
            indent=2,
        )

    if FINAL_ROOT.exists():
        previous_backup = FINAL_ROOT.with_name(
            FINAL_ROOT.name
            + ".previous."
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        shutil.move(
            FINAL_ROOT,
            previous_backup,
        )

        print(
            "Previous final result directory preserved at:\n"
            f"{previous_backup}"
        )

    shutil.move(
        BUILD_ROOT,
        FINAL_ROOT,
    )

    print()
    print("=" * 100)
    print("FINAL SKIP-AWARE MERGE COMPLETE")
    print("=" * 100)
    print("Planned configurations: 96")
    print("Completed configurations: 94")
    print("Unsupported configurations: 2")
    print("Completed fold rows: 470")
    print("Folds per completed configuration: 5")
    print()
    print(f"Final output directory:\n{FINAL_ROOT}")
    print()
    print("BEST RESULTS")
    print("-" * 100)

    print(
        best[
            [
                "scenario_display",
                "feature",
                "model_display",
                "image_macro_f1",
                "subgroup_macro_f1",
                "macro_f1_improvement_pp",
                "image_accuracy",
                "subgroup_accuracy",
                "accuracy_improvement_pp",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
