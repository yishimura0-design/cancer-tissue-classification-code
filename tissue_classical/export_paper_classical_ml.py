#!/usr/bin/env python3
from __future__ import annotations
import os

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(os.environ.get("CANCER_DATA_ROOT", "./private_data"))
FINAL = ROOT / 'ml_results_cancer_tissue_all8_selected6_final'
OLD_ROOT = ROOT / 'ml_results_cancer_tissue_grouped_clean_stage1_5fold'
NEW_ROOT = ROOT / 'ml_results_cancer_tissue_missing4_selected6_parallel'
QC_ROOT = ROOT / 'qc_outputs_clean'
MANIFEST = QC_ROOT / 'manifest_accepted.csv'
OUT = FINAL / 'paper_classical_ml_information'
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = FINAL / 'ALL_94_COMPLETED_CONFIGURATION_RESULTS.csv'
FOLD_FILE = FINAL / 'ALL_470_FOLD_RESULTS.csv'
ACCOUNTING_FILE = FINAL / 'CONFIGURATION_ACCOUNTING_96.csv'
SKIP_FILE = FINAL / 'ALL_skipped_or_failed_configurations.csv'

FEATURES = [
    'raw_stride8_gray',
    'raw_stride16_gray',
    'resize64_gray',
    'resize128_gray',
    'resize64_rgb',
    'hist_gray64',
    'hist_rgb32',
    'hog128_gray',
]
MODELS = [
    'ExtraTrees',
    'RandomForest',
    'LinearSVC',
    'SGD_LogLoss',
    'DecisionTree',
    'LightGBM',
]
MODEL_DISPLAY = {
    'ExtraTrees': 'Extra Trees',
    'RandomForest': 'Random Forest',
    'LinearSVC': 'Linear Support Vector Machine',
    'SGD_LogLoss': 'Stochastic Gradient Descent with logistic loss',
    'DecisionTree': 'Decision Tree',
    'LightGBM': 'LightGBM',
}
SCENARIOS = ['scenario1', 'scenario2']
SCENARIO_DISPLAY = {'scenario1': 'Scenario 1', 'scenario2': 'Scenario 2'}
CLASS_ORDER = {
    'scenario1': ['Normal', 'Malignant'],
    'scenario2': ['Normal', 'Stage IB', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IV'],
}
FALLBACK_CLASS_IMAGE_COUNTS = {
    'scenario1': {'Normal': 2160, 'Malignant': 14040},
    'scenario2': {
        'Normal': 2160,
        'Stage IB': 2520,
        'Stage IIA': 7920,
        'Stage IIB': 2160,
        'Stage III': 720,
        'Stage IV': 720,
    },
}
EXPECTED_IMAGES_PER_SUBGROUP = 72

MISSING: list[str] = []


def require(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Required file missing or empty: {path}')


def norm(value: Any) -> str:
    return re.sub(r'[^a-z0-9]+', '', str(value).strip().lower())


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    mapping = {norm(c): c for c in df.columns}
    for candidate in candidates:
        key = norm(candidate)
        if key in mapping:
            return mapping[key]
    return None


def parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def fmt_mean_std(mean_value: Any, std_value: Any) -> str:
    if pd.isna(mean_value):
        return 'NOT CALCULATED'
    if pd.isna(std_value):
        return f'{float(mean_value):.4f}'
    return f'{float(mean_value):.4f} ± {float(std_value):.4f}'


def fmt_value(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return 'NOT CALCULATED'
    return f'{float(value):.{digits}f}'


def metric_pair(df: pd.DataFrame, mean_candidates: list[str], std_candidates: list[str]) -> tuple[str | None, str | None]:
    return find_col(df, mean_candidates), find_col(df, std_candidates)


def get_metric_columns(summary: pd.DataFrame) -> dict[str, tuple[str | None, str | None]]:
    return {
        'image_accuracy': metric_pair(summary, ['cv_image_acc_mean', 'image_accuracy_mean'], ['cv_image_acc_std', 'image_accuracy_std']),
        'image_balanced_accuracy': metric_pair(summary, ['cv_image_balanced_acc_mean', 'cv_image_balanced_accuracy_mean', 'image_balanced_accuracy_mean'], ['cv_image_balanced_acc_std', 'cv_image_balanced_accuracy_std', 'image_balanced_accuracy_std']),
        'image_macro_f1': metric_pair(summary, ['cv_image_macro_f1_mean', 'image_macro_f1_mean'], ['cv_image_macro_f1_std', 'image_macro_f1_std']),
        'image_weighted_f1': metric_pair(summary, ['cv_image_weighted_f1_mean', 'image_weighted_f1_mean'], ['cv_image_weighted_f1_std', 'image_weighted_f1_std']),
        'group_accuracy': metric_pair(summary, ['cv_group_acc_mean', 'cv_subgroup_acc_mean', 'subgroup_accuracy_mean'], ['cv_group_acc_std', 'cv_subgroup_acc_std', 'subgroup_accuracy_std']),
        'group_balanced_accuracy': metric_pair(summary, ['cv_group_balanced_acc_mean', 'cv_group_balanced_accuracy_mean', 'cv_subgroup_balanced_acc_mean'], ['cv_group_balanced_acc_std', 'cv_group_balanced_accuracy_std', 'cv_subgroup_balanced_acc_std']),
        'group_macro_f1': metric_pair(summary, ['cv_group_macro_f1_mean', 'cv_subgroup_macro_f1_mean', 'subgroup_macro_f1_mean'], ['cv_group_macro_f1_std', 'cv_subgroup_macro_f1_std', 'subgroup_macro_f1_std']),
        'group_weighted_f1': metric_pair(summary, ['cv_group_weighted_f1_mean', 'cv_subgroup_weighted_f1_mean'], ['cv_group_weighted_f1_std', 'cv_subgroup_weighted_f1_std']),
    }


def accepted_manifest_info() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    info: dict[str, Any] = {
        'manifest_path': str(MANIFEST),
        'manifest_rows': None,
        'sample_ids': None,
        'subgroups': None,
        'excluded_files': None,
        'sample_column': None,
        'group_column': None,
    }
    if not MANIFEST.exists():
        MISSING.append(f'A1: accepted manifest not found at {MANIFEST}')
        return None, info
    manifest = pd.read_csv(MANIFEST)
    info['manifest_rows'] = len(manifest)
    sample_col = find_col(manifest, ['sample_id', 'specimen_id', 'specimen', 'sample', 'patient_id'])
    group_col = find_col(manifest, ['subgroup_id', 'tissue_subgroup', 'group_id', 'subgroup', 'group'])
    if sample_col:
        info['sample_ids'] = int(manifest[sample_col].nunique(dropna=True))
        info['sample_column'] = sample_col
    else:
        MISSING.append('A1: specimen/sample-ID column was not found in manifest_accepted.csv')
    if group_col:
        info['subgroups'] = int(manifest[group_col].nunique(dropna=True))
        info['group_column'] = group_col
    else:
        MISSING.append('A1: subgroup-ID column was not found in manifest_accepted.csv')

    excluded = 0
    found_exclusion_file = False
    if QC_ROOT.exists():
        for path in sorted(QC_ROOT.glob('*')):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not any(token in name for token in ['exclude', 'excluded', 'reject', 'rejected', 'quarantine', 'quarantined']):
                continue
            try:
                if path.suffix.lower() == '.csv':
                    excluded += len(pd.read_csv(path))
                    found_exclusion_file = True
                elif path.suffix.lower() in {'.json', '.jsonl'}:
                    if path.suffix.lower() == '.jsonl':
                        excluded += sum(1 for line in path.read_text(encoding='utf-8', errors='ignore').splitlines() if line.strip())
                        found_exclusion_file = True
                    else:
                        obj = json.loads(path.read_text(encoding='utf-8'))
                        if isinstance(obj, list):
                            excluded += len(obj)
                            found_exclusion_file = True
            except Exception:
                continue
    if found_exclusion_file:
        info['excluded_files'] = excluded
    else:
        MISSING.append('A1: excluded/quarantined file count was not recoverable from qc_outputs_clean; manifest_accepted.csv contains accepted files only')
    return manifest, info


def scenario_dataset_rows(summary: pd.DataFrame, manifest_info: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        s = summary[summary['scenario'] == scenario]
        if s.empty:
            continue
        first = s.iloc[0]
        n_images = int(first['n_images']) if 'n_images' in s.columns and not pd.isna(first['n_images']) else 16200
        n_subgroups = int(first['n_subgroups']) if 'n_subgroups' in s.columns and not pd.isna(first['n_subgroups']) else (manifest_info.get('subgroups') or 225)
        class_counts = parse_mapping(first.get('class_counts', np.nan))
        if not class_counts:
            class_counts = FALLBACK_CLASS_IMAGE_COUNTS[scenario]
        images_per_subgroup = n_images / n_subgroups if n_subgroups else EXPECTED_IMAGES_PER_SUBGROUP
        subgroup_counts = {label: int(round(int(class_counts.get(label, 0)) / images_per_subgroup)) for label in CLASS_ORDER[scenario]}
        rows.append({
            'Scenario': SCENARIO_DISPLAY[scenario],
            'Prediction classes': ', '.join(CLASS_ORDER[scenario]),
            'Total images': n_images,
            'Total tissue subgroups': n_subgroups,
            'Specimens/sample IDs': manifest_info.get('sample_ids') if manifest_info.get('sample_ids') is not None else 'NOT RECOVERED',
            'Class-wise images': json.dumps({k: int(class_counts.get(k, 0)) for k in CLASS_ORDER[scenario]}, ensure_ascii=False),
            'Class-wise subgroups': json.dumps(subgroup_counts, ensure_ascii=False),
            'Excluded/quarantined files': manifest_info.get('excluded_files') if manifest_info.get('excluded_files') is not None else 'NOT RECOVERED',
            'Prediction unit': 'Individual image and tissue subgroup',
        })
    return pd.DataFrame(rows)


def fold_assignment_consistency() -> str:
    records = []
    roots = [OLD_ROOT, NEW_ROOT]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*.csv'):
            if 'fold' not in path.name.lower() or 'assign' not in path.name.lower():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            scenario_col = find_col(df, ['scenario', 'scenario_name'])
            group_col = find_col(df, ['subgroup_id', 'group_id', 'subgroup', 'group'])
            fold_col = find_col(df, ['fold', 'fold_id', 'fold_number'])
            feature_col = find_col(df, ['feature', 'feature_name'])
            if not scenario_col or not group_col or not fold_col:
                continue
            inferred_feature = next((f for f in FEATURES if f in str(path)), None)
            for scenario in SCENARIOS:
                sdf = df[df[scenario_col].astype(str).str.lower().eq(scenario)].copy()
                if sdf.empty:
                    continue
                if feature_col:
                    feature_values = sdf[feature_col].dropna().astype(str).unique().tolist()
                else:
                    feature_values = [inferred_feature] if inferred_feature else []
                for feature in feature_values:
                    fdf = sdf[sdf[feature_col].astype(str).eq(feature)] if feature_col else sdf
                    mapping = fdf[[group_col, fold_col]].astype(str).drop_duplicates().sort_values([group_col, fold_col])
                    digest = hashlib.sha256(mapping.to_csv(index=False).encode('utf-8')).hexdigest()
                    records.append((scenario, feature, digest, str(path)))
    if not records:
        MISSING.append('A2: exact identical-fold assignment could not be independently verified because no fold-assignment CSV was found')
        return 'NOT INDEPENDENTLY VERIFIED FROM FINAL OUTPUTS'
    df = pd.DataFrame(records, columns=['scenario', 'feature', 'hash', 'file'])
    messages = []
    overall_pass = True
    for scenario in SCENARIOS:
        sdf = df[df['scenario'] == scenario]
        feature_hashes = sdf.groupby('feature')['hash'].agg(lambda x: sorted(set(x)))
        if len(feature_hashes) < len(FEATURES):
            overall_pass = False
            messages.append(f'{SCENARIO_DISPLAY[scenario]}: assignment files found for {len(feature_hashes)}/{len(FEATURES)} features')
            continue
        if any(len(hashes) != 1 for hashes in feature_hashes):
            overall_pass = False
            messages.append(f'{SCENARIO_DISPLAY[scenario]}: at least one feature has multiple assignment mappings')
            continue
        unique_hashes = {hashes[0] for hashes in feature_hashes}
        if len(unique_hashes) == 1:
            messages.append(f'{SCENARIO_DISPLAY[scenario]}: PASS, identical subgroup-to-fold mapping across all eight features')
        else:
            overall_pass = False
            messages.append(f'{SCENARIO_DISPLAY[scenario]}: FAIL, mappings differ across features')
    if not overall_pass:
        MISSING.append('A2: fold-assignment consistency requires manual review; see report')
    return '; '.join(messages)


def search_space_table(accounting: pd.DataFrame) -> pd.DataFrame:
    table = accounting.copy()
    table['Scenario'] = table.get('scenario_display', table['scenario'].map(SCENARIO_DISPLAY))
    table['Classifier'] = table.get('model_display', table['model'].map(MODEL_DISPLAY))
    table['Feature representation'] = table['feature']
    table['Completed'] = table['status'].eq('completed').map({True: 'Yes', False: 'No'})
    table['Skip/failure reason'] = table['reason'].fillna('').replace('', '—')
    return table[['Scenario', 'Classifier', 'Feature representation', 'Completed', 'Skip/failure reason', 'folds_completed']].sort_values(['Scenario', 'Classifier', 'Feature representation'])


def build_performance_table(summary: pd.DataFrame, metrics: dict[str, tuple[str | None, str | None]], scenario: str) -> pd.DataFrame:
    sdf = summary[summary['scenario'] == scenario].copy()
    rows = []
    for _, row in sdf.iterrows():
        out = {
            'Classifier': MODEL_DISPLAY.get(row['model'], row['model']),
            'Feature': row['feature'],
        }
        for label, (mean_col, std_col) in metrics.items():
            out[label] = fmt_mean_std(row[mean_col], row[std_col]) if mean_col else 'NOT CALCULATED'
        rows.append(out)
    table = pd.DataFrame(rows)
    model_order = {MODEL_DISPLAY[m]: i for i, m in enumerate(MODELS)}
    feature_order = {f: i for i, f in enumerate(FEATURES)}
    table['_model_order'] = table['Classifier'].map(model_order)
    table['_feature_order'] = table['Feature'].map(feature_order)
    return table.sort_values(['_model_order', '_feature_order']).drop(columns=['_model_order', '_feature_order'])


def classifier_first_text(table: pd.DataFrame, scenario_name: str) -> str:
    lines = [scenario_name, '=' * 160]
    metric_columns = [c for c in table.columns if c not in ['Classifier', 'Feature']]
    for model in [MODEL_DISPLAY[m] for m in MODELS]:
        mdf = table[table['Classifier'] == model]
        if mdf.empty:
            continue
        lines.extend(['', f'CLASSIFIER: {model}', '-' * 160])
        lines.append(mdf[['Feature'] + metric_columns].to_string(index=False))
    return '\n'.join(lines)


def best_rows(summary: pd.DataFrame, metrics: dict[str, tuple[str | None, str | None]]) -> pd.DataFrame:
    group_macro_mean = metrics['group_macro_f1'][0]
    if not group_macro_mean:
        raise RuntimeError('Subgroup Macro-F1 mean column not found')
    return (summary.sort_values(['scenario', group_macro_mean], ascending=[True, False]).groupby('scenario', group_keys=False).head(1).reset_index(drop=True))


def fold_column_map(folds: pd.DataFrame) -> dict[str, str | None]:
    return {
        'fold': find_col(folds, ['fold', 'fold_id', 'fold_number']),
        'val_images': find_col(folds, ['validation_images', 'n_validation_images', 'n_val_images', 'test_images', 'n_test_images', 'val_images']),
        'val_groups': find_col(folds, ['validation_subgroups', 'n_validation_subgroups', 'n_val_groups', 'test_groups', 'n_test_groups', 'val_groups']),
        'image_acc': find_col(folds, ['fold_image_acc', 'image_acc', 'image_accuracy']),
        'image_macro': find_col(folds, ['fold_image_macro_f1', 'image_macro_f1']),
        'group_acc': find_col(folds, ['fold_group_acc', 'group_acc', 'subgroup_acc', 'group_accuracy']),
        'group_macro': find_col(folds, ['fold_group_macro_f1', 'group_macro_f1', 'subgroup_macro_f1']),
        'group_bal_acc': find_col(folds, ['fold_group_balanced_acc', 'group_balanced_acc', 'subgroup_balanced_accuracy']),
        'group_weighted': find_col(folds, ['fold_group_weighted_f1', 'group_weighted_f1', 'subgroup_weighted_f1']),
        'class_counts': find_col(folds, ['validation_class_counts', 'val_class_counts', 'test_class_counts', 'validation_group_class_counts']),
    }


def best_fold_table(folds: pd.DataFrame, best_row: pd.Series) -> tuple[pd.DataFrame, str]:
    fdf = folds[(folds['scenario'] == best_row['scenario']) & (folds['feature'] == best_row['feature']) & (folds['model'] == best_row['model'])].copy()
    cmap = fold_column_map(fdf)
    if not cmap['fold']:
        MISSING.append(f"A8: fold identifier not found for {SCENARIO_DISPLAY[best_row['scenario']]}")
        return pd.DataFrame(), 'NOT AVAILABLE'
    out = pd.DataFrame({'Fold': fdf[cmap['fold']]})
    fields = [
        ('Validation images', 'val_images'),
        ('Validation subgroups', 'val_groups'),
        ('Image accuracy', 'image_acc'),
        ('Image Macro-F1', 'image_macro'),
        ('Subgroup accuracy', 'group_acc'),
        ('Subgroup balanced accuracy', 'group_bal_acc'),
        ('Subgroup Macro-F1', 'group_macro'),
        ('Subgroup weighted F1', 'group_weighted'),
        ('Validation class distribution', 'class_counts'),
    ]
    for display, key in fields:
        col = cmap[key]
        out[display] = fdf[col].values if col else 'NOT CALCULATED'
    every_fold = 'NOT DETERMINED'
    if cmap['image_macro'] and cmap['group_macro']:
        increased = (pd.to_numeric(fdf[cmap['group_macro']], errors='coerce') > pd.to_numeric(fdf[cmap['image_macro']], errors='coerce'))
        every_fold = f"Subgroup Macro-F1 higher in {int(increased.sum())}/{len(increased)} folds; {'every fold' if increased.all() else 'not every fold'}"
    if cmap['class_counts']:
        missing_class_folds = []
        for _, row in fdf.iterrows():
            counts = parse_mapping(row[cmap['class_counts']])
            if counts and any(int(counts.get(label, 0)) == 0 for label in CLASS_ORDER[best_row['scenario']]):
                missing_class_folds.append(str(row[cmap['fold']]))
        if missing_class_folds:
            every_fold += f"; folds lacking at least one class: {', '.join(missing_class_folds)}"
        else:
            every_fold += '; no fold lacked a class according to available class-count field'
    else:
        MISSING.append(f"A8: validation class distribution by fold not present for {SCENARIO_DISPLAY[best_row['scenario']]}")
    return out, every_fold


def class_token(label: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', label).strip('_')


def classwise_from_summary(best_row: pd.Series, subgroup_support: dict[str, int], specificity: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for label in CLASS_ORDER[best_row['scenario']]:
        token = class_token(label)
        precision_col = f'oof_group_{token}_precision'
        recall_col = f'oof_group_{token}_recall'
        f1_col = f'oof_group_{token}_f1'
        rows.append({
            'Class': label,
            'Precision': fmt_value(best_row.get(precision_col, np.nan)),
            'Recall/sensitivity': fmt_value(best_row.get(recall_col, np.nan)),
            'F1-score': fmt_value(best_row.get(f1_col, np.nan)),
            'Support (subgroups)': subgroup_support.get(label, 'NOT RECOVERED'),
            'Specificity': fmt_value(specificity.get(label)) if specificity and label in specificity else 'NOT CALCULATED',
        })
    return pd.DataFrame(rows)


def candidate_prediction_files(source_dir: Path, scenario: str, feature: str, model: str) -> list[Path]:
    candidates = []
    if not source_dir.exists():
        return candidates
    for path in source_dir.rglob('*.csv'):
        try:
            header = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        true_col = find_col(header, ['y_true', 'true_label', 'actual_label', 'actual', 'ground_truth'])
        pred_col = find_col(header, ['y_pred', 'predicted_label', 'prediction', 'pred_label'])
        if not true_col or not pred_col:
            continue
        ptext = norm(str(path))
        # Prefer files that identify the requested model/scenario by path or columns.
        score = 0
        if norm(model) in ptext:
            score += 3
        if norm(scenario) in ptext:
            score += 2
        if norm(feature) in ptext:
            score += 1
        if 'group' in path.name.lower() or 'subgroup' in path.name.lower():
            score += 4
        candidates.append((score, path))
    return [path for _, path in sorted(candidates, key=lambda x: (-x[0], str(x[1])))]


def normalize_labels(series: pd.Series, classes: list[str]) -> pd.Series:
    s = series.copy()
    numeric = pd.to_numeric(s, errors='coerce')
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset(set(range(len(classes)))):
        mapping = {i: label for i, label in enumerate(classes)}
        return numeric.astype(int).map(mapping)
    return s.astype(str).str.strip()


def confusion_from_predictions(best_row: pd.Series) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict[str, float] | None, str]:
    source_dir = Path(str(best_row.get('source_dir', '')))
    classes = CLASS_ORDER[best_row['scenario']]
    for path in candidate_prediction_files(source_dir, best_row['scenario'], best_row['feature'], best_row['model']):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        scenario_col = find_col(df, ['scenario', 'scenario_name'])
        model_col = find_col(df, ['model', 'model_name', 'classifier'])
        feature_col = find_col(df, ['feature', 'feature_name'])
        level_col = find_col(df, ['evaluation_level', 'level', 'prediction_level', 'unit'])
        true_col = find_col(df, ['y_true', 'true_label', 'actual_label', 'actual', 'ground_truth'])
        pred_col = find_col(df, ['y_pred', 'predicted_label', 'prediction', 'pred_label'])
        if not true_col or not pred_col:
            continue
        if scenario_col:
            df = df[df[scenario_col].astype(str).str.lower().eq(best_row['scenario'])]
        elif norm(best_row['scenario']) not in norm(str(path)):
            continue
        if model_col:
            df = df[df[model_col].astype(str).map(norm).eq(norm(best_row['model']))]
        elif norm(best_row['model']) not in norm(str(path)) and source_dir == OLD_ROOT / best_row['feature']:
            continue
        if feature_col:
            df = df[df[feature_col].astype(str).eq(best_row['feature'])]
        if level_col:
            df = df[df[level_col].astype(str).str.lower().str.contains('group|subgroup', regex=True)]
        elif not ('group' in path.name.lower() or 'subgroup' in path.name.lower()):
            # Heuristic: subgroup prediction files should be small (~225 rows).
            if len(df) > 1000:
                continue
        if df.empty:
            continue
        y_true = normalize_labels(df[true_col], classes)
        y_pred = normalize_labels(df[pred_col], classes)
        valid = y_true.isin(classes) & y_pred.isin(classes)
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if y_true.empty:
            continue
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        counts = pd.DataFrame(cm, index=classes, columns=classes)
        counts.index.name = 'True class'
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        norm_df = pd.DataFrame(normalized, index=classes, columns=classes)
        norm_df.index.name = 'True class'
        specificity = {}
        total = cm.sum()
        for i, label in enumerate(classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = total - tp - fn - fp
            specificity[label] = tn / (tn + fp) if (tn + fp) else np.nan
        return counts, norm_df, specificity, str(path)
    MISSING.append(f"A7: pooled subgroup OOF prediction file was not automatically located for {SCENARIO_DISPLAY[best_row['scenario']]} best configuration")
    return None, None, None, 'NOT LOCATED'


def most_common_off_diagonal(counts: pd.DataFrame) -> str:
    arr = counts.to_numpy().copy()
    np.fill_diagonal(arr, 0)
    if arr.max() <= 0:
        return 'No off-diagonal confusion'
    i, j = np.unravel_index(np.argmax(arr), arr.shape)
    return f'{counts.index[i]} → {counts.columns[j]}: {int(arr[i, j])}'


def main() -> None:
    for path in [SUMMARY_FILE, FOLD_FILE, ACCOUNTING_FILE]:
        require(path)
    summary = pd.read_csv(SUMMARY_FILE)
    folds = pd.read_csv(FOLD_FILE)
    accounting = pd.read_csv(ACCOUNTING_FILE)
    if len(summary) != 94:
        raise RuntimeError(f'Expected 94 completed configurations, found {len(summary)}')
    if len(folds) != 470:
        raise RuntimeError(f'Expected 470 fold rows, found {len(folds)}')

    manifest, manifest_info = accepted_manifest_info()
    dataset_table = scenario_dataset_rows(summary, manifest_info)
    dataset_table.to_csv(OUT / 'A1_FINAL_EVALUATED_DATASET.csv', index=False)

    fold_consistency = fold_assignment_consistency()
    config_table = search_space_table(accounting)
    config_table.to_csv(OUT / 'A2_CONFIGURATION_STATUS_96.csv', index=False)

    metrics = get_metric_columns(summary)
    if metrics['image_balanced_accuracy'][0] is None:
        MISSING.append('A3/A5: five-fold image-level balanced accuracy mean and SD were not present in the final summary')
    if metrics['group_balanced_accuracy'][0] is None:
        MISSING.append('A3/A5: five-fold subgroup-level balanced accuracy mean and SD were not present in the final summary')

    scenario_tables = {}
    for scenario in SCENARIOS:
        table = build_performance_table(summary, metrics, scenario)
        scenario_tables[scenario] = table
        table.to_csv(OUT / f'A3_{scenario.upper()}_ALL_RESULTS_CLASSIFIER_FIRST.csv', index=False)
        (OUT / f'A3_{scenario.upper()}_ALL_RESULTS_CLASSIFIER_FIRST.txt').write_text(classifier_first_text(table, SCENARIO_DISPLAY[scenario]) + '\n', encoding='utf-8')

    best = best_rows(summary, metrics)
    best_out_rows = []
    image_vs_group_rows = []
    classwise_tables = {}
    confusion_texts = {}
    fold_tables = {}
    fold_notes = {}

    group_macro_mean, group_macro_std = metrics['group_macro_f1']
    for _, best_row in best.iterrows():
        scenario = best_row['scenario']
        best_out_rows.append({
            'Scenario': SCENARIO_DISPLAY[scenario],
            'Best feature': best_row['feature'],
            'Best classifier': MODEL_DISPLAY[best_row['model']],
            'Selection metric': 'Subgroup Macro-F1',
            'Mean ± SD': fmt_mean_std(best_row[group_macro_mean], best_row[group_macro_std]),
        })

        level_defs = [
            ('Image', 'image_accuracy', 'image_balanced_accuracy', 'image_macro_f1', 'image_weighted_f1'),
            ('Tissue subgroup', 'group_accuracy', 'group_balanced_accuracy', 'group_macro_f1', 'group_weighted_f1'),
        ]
        for level, acc_key, bal_key, macro_key, weighted_key in level_defs:
            acc_mean, acc_std = metrics[acc_key]
            bal_mean, bal_std = metrics[bal_key]
            macro_mean, macro_std = metrics[macro_key]
            weighted_mean, weighted_std = metrics[weighted_key]
            image_vs_group_rows.append({
                'Scenario': SCENARIO_DISPLAY[scenario],
                'Best feature': best_row['feature'],
                'Best classifier': MODEL_DISPLAY[best_row['model']],
                'Evaluation level': level,
                'Accuracy': fmt_mean_std(best_row[acc_mean], best_row[acc_std]) if acc_mean else 'NOT CALCULATED',
                'Balanced accuracy': fmt_mean_std(best_row[bal_mean], best_row[bal_std]) if bal_mean else 'NOT CALCULATED',
                'Macro-F1': fmt_mean_std(best_row[macro_mean], best_row[macro_std]) if macro_mean else 'NOT CALCULATED',
                'Weighted F1': fmt_mean_std(best_row[weighted_mean], best_row[weighted_std]) if weighted_mean else 'NOT CALCULATED',
            })

        # Supports are subgroup counts inferred from the clean 72-images-per-subgroup structure.
        dsrow = dataset_table[dataset_table['Scenario'] == SCENARIO_DISPLAY[scenario]].iloc[0]
        subgroup_support = parse_mapping(dsrow['Class-wise subgroups'])
        counts, norm_cm, specificity, prediction_source = confusion_from_predictions(best_row)
        classwise = classwise_from_summary(best_row, {k: int(v) for k, v in subgroup_support.items()}, specificity)
        classwise_tables[scenario] = classwise
        classwise.to_csv(OUT / f'A6_{scenario.upper()}_BEST_CLASSWISE_SUBGROUP.csv', index=False)

        if counts is not None and norm_cm is not None:
            counts.to_csv(OUT / f'A7_{scenario.upper()}_BEST_CONFUSION_COUNTS.csv')
            norm_cm.to_csv(OUT / f'A7_{scenario.upper()}_BEST_CONFUSION_ROW_NORMALIZED.csv')
            confusion_texts[scenario] = (
                f'Prediction source: {prediction_source}\n\nRAW COUNTS\n{counts.to_string()}\n\n'
                f'ROW-NORMALIZED\n{norm_cm.to_string(float_format=lambda x: f"{x:.4f}")}\n\n'
                f'Most common off-diagonal confusion: {most_common_off_diagonal(counts)}'
            )
        else:
            confusion_texts[scenario] = f'Confusion matrix could not be automatically reconstructed. Prediction source: {prediction_source}'

        ftable, fnote = best_fold_table(folds, best_row)
        fold_tables[scenario] = ftable
        fold_notes[scenario] = fnote
        if not ftable.empty:
            ftable.to_csv(OUT / f'A8_{scenario.upper()}_BEST_FOLD_VARIABILITY.csv', index=False)

    best_table = pd.DataFrame(best_out_rows)
    best_table.to_csv(OUT / 'A4_BEST_CONFIGURATION_PER_SCENARIO.csv', index=False)
    image_vs_group = pd.DataFrame(image_vs_group_rows)
    image_vs_group.to_csv(OUT / 'A5_IMAGE_VS_SUBGROUP_BEST.csv', index=False)

    # Add explicit absolute differences for the best rows.
    diff_rows = []
    for _, best_row in best.iterrows():
        scenario = best_row['scenario']
        for metric_name, image_key, group_key in [
            ('Accuracy', 'image_accuracy', 'group_accuracy'),
            ('Balanced accuracy', 'image_balanced_accuracy', 'group_balanced_accuracy'),
            ('Macro-F1', 'image_macro_f1', 'group_macro_f1'),
            ('Weighted F1', 'image_weighted_f1', 'group_weighted_f1'),
        ]:
            im_col = metrics[image_key][0]
            gr_col = metrics[group_key][0]
            if im_col and gr_col:
                difference = float(best_row[gr_col]) - float(best_row[im_col])
                diff_rows.append({
                    'Scenario': SCENARIO_DISPLAY[scenario],
                    'Metric': metric_name,
                    'Image mean': float(best_row[im_col]),
                    'Subgroup mean': float(best_row[gr_col]),
                    'Absolute difference': difference,
                    'Difference in percentage points': difference * 100,
                })
            else:
                diff_rows.append({
                    'Scenario': SCENARIO_DISPLAY[scenario],
                    'Metric': metric_name,
                    'Image mean': np.nan,
                    'Subgroup mean': np.nan,
                    'Absolute difference': np.nan,
                    'Difference in percentage points': np.nan,
                })
    diff_table = pd.DataFrame(diff_rows)
    diff_table.to_csv(OUT / 'A5_IMAGE_VS_SUBGROUP_DIFFERENCES.csv', index=False)

    master = []
    add = master.append
    add('CLASSICAL MACHINE-LEARNING RESULTS: COMPLETE PAPER INFORMATION')
    add('=' * 180)
    add('')
    add('A1. FINAL EVALUATED DATASET')
    add('-' * 180)
    add(dataset_table.to_string(index=False))
    add('')
    add(f"Accepted manifest: {manifest_info['manifest_path']}")
    add(f"Accepted manifest rows: {manifest_info['manifest_rows']}")
    add(f"Detected subgroup column: {manifest_info['group_column']}")
    add(f"Detected specimen/sample column: {manifest_info['sample_column']}")
    add('')

    add('A2. EXACT CLASSICAL ML SEARCH SPACE')
    add('-' * 180)
    add('Features evaluated: ' + ', '.join(FEATURES))
    add('Classifiers retained: ' + ', '.join(MODEL_DISPLAY[m] for m in MODELS))
    add('Planned feature-classifier pairs: 48')
    add('Completed feature-classifier pairs: 47')
    add('Computationally unsupported pairs: 1')
    add('Planned scenario-specific configurations: 96')
    add('Completed scenario-specific configurations: 94')
    add('Computationally unsupported scenario-specific configurations: 2')
    add('Primary configuration-selection metric: mean subgroup-level Macro-F1 across five folds')
    add('Fold-assignment verification: ' + fold_consistency)
    add('')
    add('Configuration status table (classifier first, feature second):')
    add(config_table.to_string(index=False))
    add('')

    add('A3. COMPLETE OVERALL PERFORMANCE')
    add('-' * 180)
    for scenario in SCENARIOS:
        add('')
        add(classifier_first_text(scenario_tables[scenario], SCENARIO_DISPLAY[scenario]))
    add('')
    add('NOTE: “NOT CALCULATED” means that a requested metric was not present in the generated five-fold summary; it was not replaced with zero.')
    add('')

    add('A4. BEST CONFIGURATION FOR EACH SCENARIO')
    add('-' * 180)
    add(best_table.to_string(index=False))
    same_winner = len(set(zip(best['feature'], best['model']))) == 1
    add('')
    add('Same feature-classifier winner in both scenarios: ' + ('Yes' if same_winner else 'No'))
    add('')

    add('A5. IMAGE-LEVEL VERSUS SUBGROUP-LEVEL PERFORMANCE')
    add('-' * 180)
    add(image_vs_group.to_string(index=False))
    add('')
    add('Absolute differences:')
    add(diff_table.to_string(index=False))
    add('')
    for scenario in SCENARIOS:
        add(f"{SCENARIO_DISPLAY[scenario]} fold-level aggregation check: {fold_notes[scenario]}")
    add('')
    add('Interpretation wording: use “increased,” “was higher,” or “was associated with improved performance.” Do not use “significantly improved” unless a statistical test is added.')
    add('')

    add('A6. CLASS-WISE SUBGROUP RESULTS FOR EACH BEST CONFIGURATION')
    add('-' * 180)
    for scenario in SCENARIOS:
        brow = best[best['scenario'] == scenario].iloc[0]
        add('')
        add(f"{SCENARIO_DISPLAY[scenario]} — {brow['feature']} + {MODEL_DISPLAY[brow['model']]}")
        add(classwise_tables[scenario].to_string(index=False))
    add('')

    add('A7. POOLED OUT-OF-FOLD SUBGROUP CONFUSION MATRICES')
    add('-' * 180)
    for scenario in SCENARIOS:
        add('')
        add(SCENARIO_DISPLAY[scenario])
        add(confusion_texts[scenario])
    add('')

    add('A8. FOLD VARIABILITY FOR EACH BEST CONFIGURATION')
    add('-' * 180)
    for scenario in SCENARIOS:
        brow = best[best['scenario'] == scenario].iloc[0]
        add('')
        add(f"{SCENARIO_DISPLAY[scenario]} — {brow['feature']} + {MODEL_DISPLAY[brow['model']]}")
        if fold_tables[scenario].empty:
            add('Fold table not available.')
        else:
            add(fold_tables[scenario].to_string(index=False))
            cmap = fold_column_map(folds)
            for display, key in [('Subgroup accuracy', 'group_acc'), ('Subgroup Macro-F1', 'group_macro')]:
                col = cmap[key]
                bf = folds[(folds['scenario'] == scenario) & (folds['feature'] == brow['feature']) & (folds['model'] == brow['model'])]
                if col:
                    vals = pd.to_numeric(bf[col], errors='coerce').dropna()
                    if not vals.empty:
                        add(f"{display}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}, sample SD={vals.std(ddof=1):.4f}")
        add('Fold note: ' + fold_notes[scenario])
    add('')

    add('MISSING OR NOT AUTOMATICALLY RECOVERED INFORMATION')
    add('-' * 180)
    if MISSING:
        for i, item in enumerate(dict.fromkeys(MISSING), start=1):
            add(f'{i}. {item}')
    else:
        add('None.')
    add('')
    add('OUTPUT DIRECTORY')
    add('-' * 180)
    add(str(OUT))

    master_path = OUT / 'CLASSICAL_ML_ALL_INFORMATION.txt'
    master_path.write_text('\n'.join(master) + '\n', encoding='utf-8')
    (OUT / 'MISSING_INFORMATION.txt').write_text('\n'.join(dict.fromkeys(MISSING)) + ('\n' if MISSING else 'None.\n'), encoding='utf-8')

    print('=' * 100)
    print('CLASSICAL ML PAPER INFORMATION EXPORT COMPLETE')
    print('=' * 100)
    print(f'Master TXT: {master_path}')
    print(f'Scenario 1 table: {OUT / "A3_SCENARIO1_ALL_RESULTS_CLASSIFIER_FIRST.txt"}')
    print(f'Scenario 2 table: {OUT / "A3_SCENARIO2_ALL_RESULTS_CLASSIFIER_FIRST.txt"}')
    print(f'Missing-information check: {OUT / "MISSING_INFORMATION.txt"}')


if __name__ == '__main__':
    main()
