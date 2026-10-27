# BF, ED and BF+ED — grouped five-fold classical ML

**Target:** classify `HPNE`, `PANC`, or `MIA` using matched BF and ED image pairs.

**Source:** `bfed_rawpix_3fold_cv_testsize.py`. The historical filename reflects an earlier default of 3 folds. The retained paper experiment uses `--cv 5 --cv-type group --eval-mode kfold --seed 42 --ed-norm none`. No early-stopping checkpoint selection is performed.

## Input without distributing data

Set `CANCER_DATA_ROOT` to a local folder containing `cancer_combine_new/`; organize authorized images using:

```text
CANCER_DATA_ROOT/
  cancer_combine_new/
    New_HPNE_Bf/New_HPNE_Bf/HPNE_BrightField01.tiff
    New_HPNE_ED_Result/New_HPNE_ED_Result/HPNE_ED0001.tiff
    New_PANC_Bf/New_PANC_Bf/PANC_BrightField01.tiff
    New_PANC_ED_Result/New_PANC_ED_Images/PANC_ED0001.tiff
    New_MIA_Bf/New_MIA_Bf/MIA_BrightField01.tiff
    New_MIA_ED_Result/New_MIA_ED_Result/MIA_ED0001.tiff
```

Filenames above are **schematic examples**, not distributed source images. The script pairs by class + numeric identifier, skips unmatched files, and groups any repeated pair suffixes under the same key. Original dataset: 123 matched keys (HPNE 40, PANC 40, MIA 43).

## Reproduce the completed model search

Set `CANCER_DATA_ROOT` in your shell. For each stride `8`, `16`, `32`, run:

```bash
python experiment3_bf_ed_cells/bf_ed_classical/bfed_rawpix_3fold_cv_testsize.py \
  --cv 5 --cv-type group --eval-mode kfold --seed 42 \
  --stride 16 --ed-norm none \
  --modes BFplusED,BF,ED \
  --models ExtraTrees,RandomForest,LinearSVC,SGD_LogLoss,DecisionTree,LightGBM \
  --no-final-model --out-dir OUTPUT_DIR_OUTSIDE_GIT
```

`BFplusED` means `np.concatenate([BF_raw_pixel_features, ED_raw_pixel_features], axis=1)`. It does **not** produce one merged image. Retained comparison: 3 input modes × 3 strides × 6 classifiers = 54 configurations; all evaluated with grouped five-fold CV. Report **all models**, not only the winner.

**One-fold repeat check:** LightGBM / BF+ED / stride 16 / Fold 4: accuracy 0.625 exactly matched historical; macro-F1 0.621078 vs 0.626984 historical. Group-key overlap zero.

**Limitations:** A filename-derived group ID is not proof of independent biological replication. Re-run only with authorized data. The output folder may contain per-pair identifiers—keep it private.
