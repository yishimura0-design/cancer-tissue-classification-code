# Tissue classical ML — 225-subgroup five-fold CV

**Separate from the newer 45-core CNN experiment.** Historical tissue classical ML uses 16,200 accepted image frames and **225 tissue subgroups/FOVs** as the split/prediction unit. Two targets:

1. `scenario1`: Normal / Malignant.
2. `scenario2`: Normal / Stage IB / Stage IIA / Stage IIB / Stage III / Stage IV.

## Source files in execution order

* `01_qc_make_manifest.py` — generate the accepted image manifest from an **authorized local directory**.
* `03_train_cancer_searchspace_grouped.py` — full group-disjoint five-fold search over eight image-feature types and six classifiers, keeping class labels and image/subgroup metrics.
* `04_merge_all8_selected6_skip_aware.py` — original multi-job accounting / aggregation (96 planned, 94 completed, 2 unsupported); directory root read from `CANCER_DATA_ROOT`.
* `05_regenerate_selected_best_oof_v6.py` — selected-best OOF regeneration: Scenario 1 `hist_rgb32 + ExtraTrees`, Scenario 2 `hist_rgb32 + LinearSVC`. Reconstructed 225 held-out subgroup predictions per scenario.
* `export_paper_classical_ml.py` — final manuscript-oriented tables. Sensitive per-sample exports must stay **outside this repo**.

## Minimal commands (requires data owner permission)

```bash
python tissue_classical/01_qc_make_manifest.py \
  --root /PATH/TO/PRIVATE/TISSUE_IMAGES \
  --out /PATH/OUTSIDE/GIT/qc

python tissue_classical/03_train_cancer_searchspace_grouped.py \
  --manifest /PATH/OUTSIDE/GIT/qc/manifest_accepted.csv \
  --scenarios scenario1,scenario2 --features all --models all \
  --folds 5 --seed 42 \
  --out /PATH/OUTSIDE/GIT/tissue_classical_cv
```

The main training script accepts a path-based manifest including `filepath`, `scenario1_label`, `scenario2_label` and subgroup metadata. The original QC must be rerun with the same exclusion rules to match the manuscript.

**Do not report these 225 groups as 225 independent biological cores**; the new CNN core experiment instead uses 45 biological cores. Do not publish per-image manifest, individual predictions or unredacted source path records.
