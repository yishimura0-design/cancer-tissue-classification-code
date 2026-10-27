#!/usr/bin/env python3
"""Leakage-safe core-level outer 5-fold Classical ML with INNER model selection.

Per outer fold and scenario: choose feature/estimator using only distinct
inner-validation CORES, retrain selected configuration on all non-test CORES,
then produce one out-of-fold prediction per held-out CORE. When validation
is empty, accept exactly one prespecified configuration (no model selection).

Do not describe this as patient-independent external validation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, confusion_matrix, f1_score, accuracy_score
from core_eval import class_support_metrics
from run_contract import contract, require_new_or_matching

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('classical_core_module',HERE/'03_train_cancer_searchspace_core_cv.py')
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def aggregate_by_core(test_df, aligned_scores, class_order):
    scores=pd.DataFrame(aligned_scores, columns=[f'score_{c.safe_name(z)}' for z in class_order])
    scores['core_id']=test_df['sample_id_from_folder'].astype(str).to_numpy()
    scores['y_true']=test_df['_y'].astype(str).to_numpy()
    out=[]
    names=[f'score_{c.safe_name(z)}' for z in class_order]
    for cid, rows in scores.groupby('core_id',sort=True):
        truth=rows.y_true.unique()
        if len(truth)!=1 or len(rows)!=360:
            raise ValueError(f'Core {cid} has mixed label or wrong image count')
        avg=rows[names].mean(axis=0).to_numpy(dtype=float)
        if np.isnan(avg).any():
            raise ValueError('NaN model scores')
        result={'core_id':cid,'y_true':truth[0],'y_pred':class_order[int(avg.argmax())]}
        result.update({k:float(v) for k,v in zip(names,avg)})
        out.append(result)
    return pd.DataFrame(out)


def fit_predict_core(X, sdf, ti, pi, feature_name, model_name, class_order, seed, args):
    """Fit using only `ti`, predict all heldout IMAGES then aggregate to Core."""
    model,_=c.build_model_pipeline(model_name,seed,args.n_estimators,args.max_iter,X.shape[1],args.scale_dim_limit)
    y=sdf['_y'].astype(str).to_numpy()
    core_ids=sdf['sample_id_from_folder'].astype(str).to_numpy()
    c.fit_model(model,X[ti],y[ti],core_ids[ti])
    aligned,score_type=c.image_score_matrix(model,X[pi],class_order)
    part=sdf.iloc[pi].reset_index(drop=True)
    cols=[f'score_{c.safe_name(z)}' for z in class_order]
    image=pd.DataFrame(aligned,columns=cols)
    image.insert(0,'path',part['filepath'].astype(str).to_numpy())
    image.insert(1,'core_id',part['sample_id_from_folder'].astype(str).to_numpy())
    image.insert(2,'fov_id',part['subgroup_id'].astype(str).to_numpy())
    image.insert(3,'y_true',part['_y'].astype(str).to_numpy())
    image.insert(4,'y_pred',[class_order[i] for i in aligned.argmax(axis=1)])
    image['score_type']=score_type
    if len(image)!=len(part) or image.path.duplicated().any():
        raise ValueError('Missing or duplicate image predictions')
    core=aggregate_by_core(part,aligned,class_order)
    return core,score_type,image

def validation_metric(core_df, labels):
    # Average recall across CLASSES ACTUALLY PRESENT in this inner validation.
    # Avoid inventing positives for a class with zero validation cores.
    present=sorted(set(core_df.y_true))
    if not present: return float('-inf')
    return float(recall_score(core_df.y_true,core_df.y_pred,labels=present,average='macro',zero_division=0))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--roles',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--experiment-id',default='B0')
    p.add_argument('--folds',default='1,2,3,4,5',help='For A1 execute ONLY folds 2,4')
    p.add_argument('--feature-cache-dir',default=None,help='Share deterministic per-image features across B0 and A1')
    p.add_argument('--scenarios',default='scenario1,scenario2')
    p.add_argument('--features',default='hist_rgb32,resize64_gray')
    p.add_argument('--models',default='ExtraTrees,LinearSVC,LightGBM')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--n-estimators',type=int,default=300)
    p.add_argument('--max-iter',type=int,default=8000)
    p.add_argument('--scale-dim-limit',type=int,default=400000)
    a=p.parse_args()
    fold_ids=sorted(set(int(x) for x in a.folds.split(',') if x.strip()))
    if not fold_ids or not set(fold_ids).issubset(set(range(1,6))):
        p.error('--folds must be a nonempty subset of 1,2,3,4,5')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    feature_cache=Path(a.feature_cache_dir) if a.feature_cache_dir else out/'feature_cache'
    df=c.validate_manifest(pd.read_csv(a.manifest),True)
    roles=pd.read_csv(a.roles)
    choices=c.parse_features(a.features)
    models=c.parse_models(a.models)
    if not choices or not models:raise ValueError('No candidates selected')
    if set(roles.outer_fold)!=set(range(1,6)):
        raise ValueError('Exactly five outer folds required')
    if 'core_id' not in roles or 'role' not in roles:raise ValueError('Invalid roles CSV')
    if not all((roles[roles.outer_fold==f].role=='val').any() for f in fold_ids):
        raise ValueError('Nonempty validation required in each requested fold')
    # Inner validation is mandatory for a leakage-safe candidate comparison.
    if not (roles.role=='val').any():
        raise ValueError('Inner validation cannot be empty; never silently select fixed candidates')
    records=[];image_records=[];selection=[];fold_summaries=[]
    for sc in (name.strip() for name in a.scenarios.split(',') if name.strip()):
        if sc not in c.SCENARIOS:raise ValueError(sc)
        labels=c.SCENARIOS[sc]['classes']; label_col=c.SCENARIOS[sc]['label_col']
        sdf=df[df[label_col].isin(labels)].reset_index(drop=True).copy()
        sdf['_y']=sdf[label_col].astype(str)
        if len(sdf)!=16200:raise ValueError('Unexpected scenario filtering')
        for fold in fold_ids:
            part=roles[roles.outer_fold==fold]
            if len(part)!=45 or part.core_id.nunique()!=45:
                raise ValueError(f'Bad core roles in fold {fold}')
            def indices(role_names):
                keep=set(part[part.role.isin(role_names)].core_id)
                return np.flatnonzero(sdf.sample_id_from_folder.astype(str).isin(keep).to_numpy())
            train_idx=indices(['train']); val_idx=indices(['val']); test_idx=indices(['test'])
            if not len(train_idx) or not len(val_idx) or not len(test_idx):raise ValueError('Empty train/val/test is forbidden')
            def core_set(ii):return set(sdf.iloc[ii].sample_id_from_folder)
            if core_set(train_idx)&core_set(test_idx) or core_set(val_idx)&core_set(test_idx) or core_set(train_idx)&core_set(val_idx):
                raise RuntimeError(f'BIOLOGICAL CORE LEAKAGE fold {fold}')
            if set(sdf.iloc[train_idx]._y)!=set(labels):raise ValueError('Training missing class')
            paths=sdf.filepath.astype(str).tolist()
            best=None;cache={}
            for fn,feature_spec in choices:
                X,_=c.extract_feature_matrix(paths,fn,feature_spec,feature_cache,True)
                cache[fn]=X
                for m in models:
                    skip=c.should_skip_model(m,len(train_idx),X.shape[1])
                    if skip:
                        selection.append({'scenario':sc,'fold':fold,'feature':fn,'model':m,'status':'skipped','reason':skip})
                        continue
                    score=None
                    test_pred,_,_=fit_predict_core(X,sdf,train_idx,val_idx,fn,m,labels,a.seed+fold,a)
                    score=validation_metric(test_pred,labels)
                    selection.append({'scenario':sc,'fold':fold,'feature':fn,'model':m,
                                      'status':'candidate','inner_val_macro_recall_present_classes':score})
                    rank=(score if np.isfinite(score) else 0.,fn,m)
                    # Deterministic tie handling. Never consult outer test for candidate choice.
                    if best is None or rank[0]>best[0][0]:
                        best=(rank,fn,m)
            if best is None:raise RuntimeError('All model configurations skipped')
            _,fn,m=best
            X=cache[fn]
            all_non_test=indices(['train','val'])
            # Fit final selected model on all 36 non-test cores; do not touch test in fitting.
            pred,score_type,image_pred=fit_predict_core(X,sdf,all_non_test,test_idx,fn,m,labels,a.seed+fold+1000,a)
            pred.insert(0,'fold',fold)
            pred.insert(0,'experiment_id',a.experiment_id)
            image_pred.insert(0,'fold',fold)
            image_pred.insert(0,'experiment_id',a.experiment_id)
            image_pred['scenario']=sc;image_pred['chosen_feature']=fn;image_pred['chosen_model']=m
            if len(image_pred)!=len(test_idx) or image_pred.core_id.nunique()!=9:
                raise ValueError('Incomplete held-out image predictions for fold')
            image_records.extend(image_pred.to_dict('records'))
            fold_summaries.append({'experiment_id':a.experiment_id,'scenario':sc,'fold':fold,
                'feature':fn,'model':m,'score_type':score_type,
                'n_train_cores':len(core_set(train_idx)),'n_val_cores':len(core_set(val_idx)),
                'n_test_cores':len(pred),'n_test_images':len(image_pred),
                'test_core_accuracy':float(accuracy_score(pred.y_true,pred.y_pred)),
                'test_core_macro_f1_all_classes':float(f1_score(pred.y_true,pred.y_pred,labels=labels,average='macro',zero_division=0)),
                'test_per_class':class_support_metrics(pd.Categorical(pred.y_true,categories=labels).codes,
                                     pd.Categorical(pred.y_pred,categories=labels).codes,labels)})
            pred['scenario']=sc;pred['chosen_feature']=fn;pred['chosen_model']=m
            pred['selection_mode']='inner_core_validation'
            records.extend(pred.to_dict('records'))
            selection.append({'scenario':sc,'fold':fold,'feature':fn,'model':m,'status':'selected',
                              'inner_val_macro_recall_present_classes':best[0][0]})
            print(f'[{sc}] fold={fold}: selected {fn}/{m}; inner_val_cores={len(core_set(val_idx))}, test={len(pred)}',flush=True)
            del cache
    oof=pd.DataFrame(records)
    images=pd.DataFrame(image_records)
    oof.to_csv(out/'selected_core_oof.csv',index=False)
    images.to_csv(out/'selected_image_oof.csv.gz',index=False,compression='gzip')
    pd.DataFrame(selection).to_csv(out/'inner_model_selections.csv',index=False)
    pd.DataFrame(fold_summaries).to_json(out/'fold_metrics.json',orient='records',indent=2)
    summaries=[]
    for sc,g in oof.groupby('scenario'):
        expected=9*len(fold_ids)
        ig=images[images.scenario==sc]
        if len(g)!=expected or g.core_id.nunique()!=expected:
            raise RuntimeError(f'Not exactly one Test result per {expected} cores')
        if len(ig)!=360*expected or ig.path.nunique()!=360*expected:
            raise RuntimeError('Incomplete single-image OOF coverage')
        labels=c.SCENARIOS[sc]['classes']
        summaries.append({'experiment_id':a.experiment_id,'scenario':sc,
                          'is_complete_5_fold_oof':set(fold_ids)==set(range(1,6)),
                          'n_core_predictions':len(g),'n_image_predictions':len(ig),
                          'core_accuracy':float(accuracy_score(g.y_true,g.y_pred)),
                          'pooled_core_macro_f1_all_classes':float(f1_score(g.y_true,g.y_pred,labels=labels,average='macro',zero_division=0)),
                          'per_class_core_support':{z:int((g.y_true==z).sum()) for z in labels},
                          'confusion_matrix_counts':confusion_matrix(g.y_true,g.y_pred,labels=labels).tolist(),
                          'model_selection':'inner validation per outer fold'})
    (out/'partial_or_pooled_metrics.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(f'COMPLETE: {len(fold_ids)} independent outer Test folds / {a.experiment_id}; '
          f'full 45-core pooled score only when all five folds are present.')

if __name__=='__main__':main()
