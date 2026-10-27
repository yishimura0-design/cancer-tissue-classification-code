# Experiment 3 — Pancreatic cell lines from brightfield (BF) and elliptical dichroism (ED) images

## What this experiment answers

Does ED imaging add information to conventional brightfield microscopy for telling the three cell lines apart — **HPNE**, **PANC-1** and **MIA PaCa-2**?

## The data

| Item | Value |
|---|---|
| Matched BF + ED image pairs | **123**: HPNE 40, PANC-1 40, MIA PaCa-2 43 |
| Pairing rule | Same cell line and same number, e.g. `HPNE_BrightField07` ↔ `HPNE_ED0007` |

## How the evaluation works

* Three inputs are compared: **BF only**, **ED only**, and **BF+ED** (the BF and ED pixel values placed one after the other in a single feature vector; not a merged image).
* Pixels are sampled every **8, 16 or 32** pixels from the original-resolution images (no resizing or cropping).
* Six classical classifiers: Extra Trees, Random Forest, Linear SVC, SGD (logistic), Decision Tree, LightGBM.
* Five-fold cross-validation, stratified by cell line, with each image pair kept within one fold (seed 42). **The same five folds are used for every input, sampling step and classifier**, so all comparisons are made on identical splits.
* Results are reported as **mean ± standard deviation across the five folds**.
* 3 inputs × 3 sampling steps × 6 classifiers = 54 configurations × 5 folds = 270 evaluations.

No CNN is used in this experiment.

## Steps — run from the repository root (CPU)

```bash
export CANCER_DATA_ROOT="/AUTHORIZED_PRIVATE/ROOT_CONTAINING_cancer_combine_new"
export RESULTS_ROOT="/AUTHORIZED_PRIVATE_OUTPUT/cancer_results"

for stride in 8 16 32; do
  python experiment3_bf_ed_cells/bf_ed_classical/bfed_rawpix_3fold_cv_testsize.py \
    --cv 5 --cv-type group --eval-mode kfold --seed 42 \
    --stride "$stride" --ed-norm none \
    --modes BFplusED,BF,ED \
    --models ExtraTrees,RandomForest,LinearSVC,SGD_LogLoss,DecisionTree,LightGBM \
    --no-final-model --out-dir "$RESULTS_ROOT/bfed/stride_$stride"
done
```

**Never omit `--cv 5`.** The script name and its default still refer to an earlier three-fold version; without `--cv 5` the paper is not reproduced.

Expected folder layout under `CANCER_DATA_ROOT` is described in [`bf_ed_classical/README.md`](bf_ed_classical/README.md).

## Outputs

| File | Content |
|---|---|
| `TABLE2_cv_summary_by_mode_model.csv` | Mean ± SD accuracy and macro-F1 for every input × classifier (the manuscript BF/ED numbers) |
| `cv_fold_results_long.csv` | Every fold result |
| `cv_per_class_results_long.csv` | Per-cell-line precision, recall and F1 per fold |
| `meta.json` | Settings used (folds, seed, stride, grouping) |

## What computer is needed

Ordinary CPU.

## Files in this folder

| File | Role |
|---|---|
| `bf_ed_classical/bfed_rawpix_3fold_cv_testsize.py` | Pairing, pixel features, cross-validation, summary tables |
| `bf_ed_classical/README.md` | Input folder layout and the original repeat check |
