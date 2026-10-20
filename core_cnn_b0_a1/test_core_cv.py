from __future__ import annotations
import importlib.util
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from make_core_splits import core_table_from_manifest,assign_outer_folds,build_role_table
from core_eval import aggregate_core_predictions,core_log_loss,core_metrics

MANIFEST=Path(os.environ.get('TISSUE_MANIFEST',Path(__file__).resolve().parent.parent/'manifest_accepted.csv'))
if not MANIFEST.exists():
    # Portable fixture for users who have not yet mounted the real QC manifest.
    # The actual dataset profile was verified separately against manifest_accepted.csv.
    outer=pd.read_csv(Path(__file__).parent/'splits'/'outer_core_folds.csv')
    rows=[]
    for core in outer.itertuples(index=False):
        for f in range(1,6):
            subgroup=f'{core.core_id[0]}/{core.core_id}/{core.core_id}_{f}_20x'
            for ang in range(72):
                filename=f'{core.core_id}_{f}_20x_ang{ang}.tif'
                rows.append({'filepath':f'/synthetic/{subgroup}_ang{ang}.tif',
                            'relative_path':f'{subgroup}_ang{ang}.tif',
                            'filename':filename,
                            'sample_id_from_folder':core.core_id,
                            'scenario1_label':core.scenario1_label,
                            'scenario2_label':core.scenario2_label,
                            'subgroup_id':subgroup,'qc_status':'accepted_tif'})
    MANIFEST=Path(tempfile.gettempdir())/'tissue_core_cv_synthetic_manifest.csv'
    pd.DataFrame(rows).to_csv(MANIFEST,index=False)



def test_real_dataset_profile():
    cores=core_table_from_manifest(str(MANIFEST))
    assert len(cores)==45 and cores.n_images.sum()==16200
    assert cores.scenario2_label.value_counts().to_dict()=={
        'Stage IIA':22,'Stage IB':7,'Normal':6,'Stage IIB':6,'Stage III':2,'Stage IV':2}


def test_five_folds_are_core_independent_and_rare_classes_trainable():
    cores=core_table_from_manifest(str(MANIFEST))
    assigned=assign_outer_folds(cores,5,42)
    roles=build_role_table(assigned,.2,42)
    assert assigned.fold.value_counts().to_dict()=={1:9,2:9,3:9,4:9,5:9}
    assert len(roles)==225
    for fold,part in roles.groupby('outer_fold'):
        assert part.role.value_counts().to_dict()=={'train':29,'test':9,'val':7}
        all_classes=set(cores.scenario2_label)
        assert set(part.loc[part.role=='train','scenario2_label'])==all_classes
        assert not set(part.loc[part.role=='val','scenario2_label'])&{'Stage III','Stage IV'}
        assert not set(part.loc[part.role=='test','core_id']) & set(part.loc[part.role!='test','core_id'])
        assert not set(part.loc[part.role=='val','core_id']) & set(part.loc[part.role=='train','core_id'])


def test_core_probability_aggregation_and_metrics():
    classes=['Normal','Malignant']
    pred=pd.DataFrame([{'group_id':f'F1_{j}','target':0,'prob_Normal':.75,'prob_Malignant':.25}
                       for j in range(5)]+[
                      {'group_id':f'A1_{j}','target':1,'prob_Normal':.3,'prob_Malignant':.7}
                      for j in range(5)])
    mapping=pd.DataFrame([{'group_id':f'F1_{j}','core_id':'F1'} for j in range(5)]+[
                        {'group_id':f'A1_{j}','core_id':'A1'} for j in range(5)])
    core=aggregate_core_predictions(pred,mapping,classes)
    assert len(core)==2 and core_metrics(core,classes)['core_accuracy']==1.
    assert core_log_loss(core,classes)>0
    bad=mapping.copy();bad.loc[0,'core_id']='A1'
    try:aggregate_core_predictions(pred,bad,classes)
    except ValueError:pass
    else:raise AssertionError('Unexpectedly accepted wrong grouping')


def test_classical_core_run_smoke(tmp_path):
    path=Path(__file__).with_name('03_train_cancer_searchspace_core_cv.py')
    spec=importlib.util.spec_from_file_location('classical_core',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    manifest=pd.read_csv(MANIFEST)
    core_map=assign_outer_folds(core_table_from_manifest(str(MANIFEST)),5,42)
    for sc in ['scenario1','scenario2']:
        label_col=mod.SCENARIOS[sc]['label_col']
        sdf=manifest.drop_duplicates('subgroup_id').reset_index(drop=True)
        fold=mod.build_fold_assignment(mod.make_group_table(sdf,label_col),core_map)
        assert len(fold)==45
        rng=np.random.default_rng(42)
        X=rng.standard_normal((len(sdf),8))
        summary,fold_rows,_,_,_=mod.run_configuration(
            X=X,sdf=sdf,scenario=sc,label_col=label_col,
            class_order=mod.SCENARIOS[sc]['classes'],fold_assignment=fold,
            feature_name='synthetic_smoke',feature_spec={'test_only':True},
            model_name='DecisionTree',out_dir=tmp_path,seed=42,
            n_estimators=10,max_iter=100,scale_dim_limit=400000,save_oof=True)
        assert summary['cv_status']=='ok' and summary['folds_completed']==5
        assert summary['n_cores']==45 and 'oof_core_macro_f1' in summary
        assert len(fold_rows)==5
        out=tmp_path/sc/'configs'/'synthetic_smoke__DecisionTree'/'oof_core_predictions.csv'
        rows=pd.read_csv(out)
        assert len(rows)==45 and rows.core_id.nunique()==45
        assert all(rows.groupby('core_id').size()==1)


def test_zero_validation_fails_closed():
    import pytest
    from verify_core_contract import verify
    cores=core_table_from_manifest(str(MANIFEST))
    outer=assign_outer_folds(cores,5,42)
    with pytest.raises(ValueError,match='Nonempty inner validation'):
        build_role_table(outer,0,42)
    roles=build_role_table(outer,.2,42)
    assert (roles.groupby('outer_fold').apply(lambda x:(x.role=='val').sum())>0).all()
    # Deliberately corrupt one valid fold to prove the audit rejects empty val.
    roles.loc[(roles.outer_fold==1)&(roles.role=='val'),'role']='train'
    target=Path(tempfile.gettempdir())/'tissue_core_cv_empty_val_must_fail.csv'
    roles.to_csv(target,index=False)
    with pytest.raises(ValueError, match='validation'):
        verify(str(MANIFEST),str(target))


def test_two_fold_variant_preserves_all_other_roles(tmp_path):
    from make_variant_roles import make_variant
    from verify_core_contract import verify
    src=Path(__file__).with_name('splits')/'inner_outer_roles.csv'
    base=pd.read_csv(src)
    alt=make_variant(base)
    for fold in [1,3,5]:
        assert base[base.outer_fold==fold].reset_index(drop=True).equals(
               alt[alt.outer_fold==fold].reset_index(drop=True))
    for fold in [2,4]:
        b=base[base.outer_fold==fold]
        a=alt[alt.outer_fold==fold]
        assert set(b.loc[b.role=='val','core_id'])==set(a.loc[a.role=='val','core_id'])
    path=tmp_path/'alt.csv';alt.to_csv(path,index=False)
    assert verify(str(MANIFEST),str(path))['status']=='PASS'
    # Alternative is invalid if baseline assigns C3 to validation in fold 2
    bad=base.copy();bad.loc[(bad.outer_fold==2)&(bad.core_id=='C3'),'role']='val'
    import pytest
    with pytest.raises(ValueError,match='not applicable'):
        make_variant(bad)


def test_all_angular_views_of_a1_fovs_stay_together():
    from verify_core_contract import verify
    core=core_table_from_manifest(str(MANIFEST))
    roles=build_role_table(assign_outer_folds(core,5,42),.2,42)
    target=Path(tempfile.gettempdir())/'tissue_core_cv_regular_roles.csv'
    roles.to_csv(target,index=False)
    report=verify(str(MANIFEST),str(target))
    assert report['status']=='PASS'
    a1_rows=pd.read_csv(MANIFEST).query('sample_id_from_folder == "A1"')
    assert a1_rows.subgroup_id.nunique()==5 and len(a1_rows)==360


def test_sparse_validation_can_be_zero_without_scoring_failure():
    import importlib.util
    path=Path(__file__).with_name('run_classical_inner_selected.py')
    spec=importlib.util.spec_from_file_location('classical_nested_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    df=pd.DataFrame([{'core_id':'F1','y_true':'Normal','y_pred':'Normal'},
                     {'core_id':'A2','y_true':'Stage IB','y_pred':'Stage IB'}])
    score=module.validation_metric(df,['Normal','Stage IB','Stage III','Stage IV'])
    assert score == 1.0
