"""Pure evaluation utilities: score predictions at the BIOLOGICAL CORE level.

Mean the five field-of-view probabilities for each core. Never count 360 angular
images as 360 independent biological samples in the primary CV score.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support


def aggregate_core_predictions(subgroup_predictions: pd.DataFrame,
                               group_to_core: pd.DataFrame,
                               class_names: list[str]) -> pd.DataFrame:
    """Convert original evaluate() subgroup predictions to core predictions.

    subgroup_predictions: group_id, target, prob_<label> for each field.
    group_to_core: group_id, core_id mapping, one record per field.
    """
    probability_columns = [f'prob_{c}' for c in class_names]
    expected = {'group_id', 'target', *probability_columns}
    if not expected.issubset(subgroup_predictions.columns):
        raise ValueError(f'Missing prediction columns: {expected-set(subgroup_predictions)}')
    if not {'group_id','core_id'}.issubset(group_to_core.columns):
        raise ValueError('Missing group_id/core_id map')
    map_df = group_to_core[['group_id','core_id']].drop_duplicates()
    if map_df.group_id.duplicated().any():
        raise ValueError('A field maps to more than one core')
    source = subgroup_predictions.merge(map_df,on='group_id',how='left',validate='one_to_one')
    if source.core_id.isna().any():
        raise ValueError('Unmapped field(s) in core aggregation')
    if (source.groupby('core_id')['target'].nunique() != 1).any():
        raise ValueError('Mixed true class in the same core')
    if (source.groupby('core_id')['group_id'].nunique() != 5).any():
        raise ValueError('Each core must contain exactly five fields of view')
    core = source.groupby('core_id',sort=True).agg({'target':'first',**{c:'mean' for c in probability_columns}}).reset_index()
    proba = core[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(proba).all():
        raise ValueError('Non-finite core probability')
    core['prediction'] = proba.argmax(axis=1)
    core['target_label'] = [class_names[i] for i in core['target']]
    core['prediction_label'] = [class_names[i] for i in core['prediction']]
    core['n_subgroups'] = 5
    return core


def core_log_loss(core_predictions: pd.DataFrame, class_names: list[str]) -> float:
    """Mean per-core NLL for *inner validation* even with zero rare-class cores."""
    y = core_predictions['target'].to_numpy(dtype=int)
    p = core_predictions[[f'prob_{x}' for x in class_names]].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError('No validation cores; disable early stopping or provide inner val')
    return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-9,1.0)).mean())


def class_support_metrics(y, pred, class_names):
    """Explicit N/A for missing true Test core support; real zero is preserved."""
    y=np.asarray(y,dtype=int);pred=np.asarray(pred,dtype=int)
    labels=list(range(len(class_names)))
    pr,re,f,_=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    rows=[]
    for i,name in enumerate(class_names):
        sup=int((y==i).sum());positive_pred=int((pred==i).sum())
        rows.append({'class':name,'support':sup,'predicted_count':positive_pred,
                     'precision':float(pr[i]) if positive_pred else None,
                     'recall':float(re[i]) if sup else None,
                     'f1':float(f[i]) if sup or positive_pred else None,
                     'recall_na_reason':None if sup else 'no_true_test_core_for_class'})
    return rows


def core_metrics(core_predictions: pd.DataFrame, class_names: list[str]) -> dict:
    y = core_predictions.target.to_numpy(dtype=int)
    pred = core_predictions.prediction.to_numpy(dtype=int)
    labels = list(range(len(class_names)))
    support = {name:int((y==i).sum()) for i,name in enumerate(class_names)}
    return {
        'core_accuracy':float(accuracy_score(y,pred)),
        'core_macro_f1_all_classes':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'n_cores':int(len(y)), 'support_by_class':support,
        'per_class':class_support_metrics(y,pred,class_names),
        'confusion_matrix_counts':confusion_matrix(y,pred,labels=labels).tolist(),
    }
