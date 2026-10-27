# Core CNN — B0 / A1, 45 biological cores

**Inputs:** 16,200 private microscope images from 45 cores (5 FOV × 72 views per core). Binary Normal/Malignant and six-class Normal/IB/IIA/IIB/III/IV targets.

**Primary unit:** biological core, not individual image. Outer five folds are disjoint by `sample_id_from_folder`. B0 is the baseline; A1 exchanges specified core fold ownership as a split-sensitivity analysis, not a separate cohort.

## Reproduction workflow (authorized data only)

```bash
python core_cnn_b0_a1/make_core_splits.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --out /PRIVATE_RESULTS/core_roles

python core_cnn_b0_a1/verify_core_contract.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --roles /PRIVATE_RESULTS/core_roles/inner_outer_roles.csv \
  --out /PRIVATE_RESULTS/core_audit.json

python core_cnn_b0_a1/run_one_core_cnn_fold_resumable.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --roles /PRIVATE_RESULTS/core_roles/inner_outer_roles.csv \
  --scenario scenario1 --model resnet18 --fold 1 \
  --experiment-id B0 --out /PRIVATE_RESULTS/core_cnn
```

Valid models: ResNet18, EfficientNet-B0, MobileNetV3-Large (ImageNet pretrained). B0 comprises 2 tasks × 3 architectures × 5 folds = 30 model-fold jobs; A1 reuses B0 outputs for unchanged folds and retrains only folds 2 and 4 (12 jobs). Validation decides epochs; **outer test does not select checkpoints**. Training code is in this folder, with resumed checkpoints and shared transforms in the adjacent modules.

Manifests must preserve `filepath`, `relative_path`, `sample_id_from_folder`, `subgroup_id`, `scenario1_label`, and `scenario2_label` consistent with the original QC. These columns and any real IDs should not be committed.

**Limitations:** The original split proves core-disjoint test data, not patient independence; certain cancer stages have very few core samples.

## Core-level classical ML and results reporting (B0/A1)

These three source scripts complement the Core CNN runner and have been retained from the original [cancer](https://github.com/yishimura0-design/cancer) study:

- `03_train_cancer_searchspace_core_cv.py`: exploratory, core-grouped five-fold feature/model sweep. Requires `--core-fold-map` produced by `make_core_splits.py`. Selecting configurations by the outer-fold result is exploratory and **must not** be presented as unbiased held-out model selection.
- `run_classical_inner_selected.py`: the B0/A1 Core Classical ML evaluation, choosing features/classifiers using inner validation only. It imports `03_train_cancer_searchspace_core_cv.py`, `core_eval.py`, and `run_contract.py` from this same folder.
- `publish_ccast_summary.py`: reads stored Core CNN and Core Classical results and writes de-identified, aggregated B0/A1 reports. This script does **not** upload raw microscopy data or push to GitHub.

Example commands (use approved local paths and outputs **outside this repository**):

```bash
# Core Classical B0: five independently evaluated outer folds
python core_cnn_b0_a1/run_classical_inner_selected.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --roles /PRIVATE_RESULTS/core_roles/inner_outer_roles.csv \
  --experiment-id B0 --folds 1,2,3,4,5 \
  --out /PRIVATE_RESULTS/advanced_core_cv_results/B0/classical

# Core Classical A1: only the two changed folds; reuse matching B0 predictions for others
python core_cnn_b0_a1/run_classical_inner_selected.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --roles /PRIVATE_RESULTS/core_roles_variant_a1/inner_outer_roles.csv \
  --experiment-id A1 --folds 2,4 \
  --out /PRIVATE_RESULTS/advanced_core_cv_results/A1/classical

# Aggregate de-identified B0 results after the required folds are completed
python core_cnn_b0_a1/publish_ccast_summary.py \
  --experiment B0 \
  --results-root /PRIVATE_RESULTS/advanced_core_cv_results \
  --baseline-roles /PRIVATE_RESULTS/core_roles/inner_outer_roles.csv \
  --variant-roles /PRIVATE_RESULTS/core_roles_variant_a1/inner_outer_roles.csv \
  --out /PRIVATE_RESULTS/report_stage/B0
```

The inner-selected Classical ML experiment uses **45 physical cores**, not the **225 FOV groups** in `tissue_classical/`. Preserve the original manifest and fold identities when reproducing published numbers. Do not commit raw images, core IDs, raw predictions, feature caches, or PBS logs. The restored code was copied from the maintained original source; reproducing the numerical scores still requires the authorized dataset and matching software environment.
