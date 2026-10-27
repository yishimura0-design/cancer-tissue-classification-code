# Experiment 1 — Pancreatic tissue: normal vs malignant, and cancer stage

## What this experiment answers

Can a model tell, from polarization-resolved microscope images of a tissue core:

1. **Binary task (`scenario1`)**: is the tissue **normal** or **malignant**?
2. **Six-class task (`scenario2`)**: is it **normal**, **Stage IB**, **IIA**, **IIB**, **III** or **IV**?

Both tasks use the same images, the same folds and the same code; only the label column differs.

## The data

| Item | Value |
|---|---|
| Tissue cores on the array | 48 |
| Cores that pass image quality control (used here) | **45**: 6 normal, 39 malignant |
| Malignant cores by stage | IB 7 · IIA 22 · IIB 6 · III 2 · IV 2 |
| Fields of view per core | 5 |
| Polarization-angle images per field of view | 72 |
| Images per core / in total | 360 / **16,200** |

All 360 images of a core share the core's label and always stay together.

## How the evaluation works

* The 45 cores are divided into **5 folds of 9 cores**, spread as evenly as possible by stage (rarest stages placed first, fixed seed 42).
* The model is trained **5 times**. Each time, one fold is the **test set** (9 cores); of the remaining 36 cores, **7 are a validation set** used only to decide when to stop training, and **29 are the training set**.
* Every core is tested **exactly once** by a model that did not use it for training or checkpoint selection. For CNNs, each fold starts with ImageNet-pretrained backbone weights and a newly initialized classification head, then is fine-tuned independently; trained weights are not transferred between folds.
* Stage III and IV cores are never placed in validation (too few), so all of them remain available for training.
* A core's prediction is the average over its 72 angles, then over its 5 fields of view.
* **B0** is the primary fold assignment. **A1** is a sensitivity variant in which one Stage III core and one Stage IIA core swap test folds (folds 2 and 4 only); it uses the same specimens.

## Steps — run from the repository root

Set the paths once (private locations, never inside this repository):

```bash
export TISSUE_ROOT="/AUTHORIZED_PRIVATE/TISSUE_SOURCE"
export RESULTS_ROOT="/AUTHORIZED_PRIVATE_OUTPUT/cancer_results"
mkdir -p "$RESULTS_ROOT"
```

**Step 1 — Image quality control → list of accepted images**

```bash
python experiment1_tissue/01_qc_make_manifest.py \
  --root "$TISSUE_ROOT" --out "$RESULTS_ROOT/tissue_qc"
```

Output: `manifest_accepted.csv` (expected 16,200 images, 45 cores) and `qc_summary.json`, which lists the excluded cores under `missing_sample_ids_after_qc`.

**Step 2 — Make the B0 folds, check them, and make A1**

```bash
python experiment1_tissue/core_cnn_b0_a1/make_core_splits.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --out "$RESULTS_ROOT/core_b0_roles"

python experiment1_tissue/core_cnn_b0_a1/verify_core_contract.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --out "$RESULTS_ROOT/core_roles_audit.json"

python experiment1_tissue/core_cnn_b0_a1/make_variant_roles.py \
  --baseline-roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --out "$RESULTS_ROOT/core_a1_roles"
```

Check that the folds are identical to the ones used in the original runs (the training logs record these SHA-256 values):

```bash
sha256sum "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" "$RESULTS_ROOT/core_a1_roles/inner_outer_roles.csv"
# B0: 1cfbfb97a3811f03f31210f4b2ff904704e58db989ae57c759b1427a3ab7ff2a
# A1: 3ef07aaa65979af22469f96014aac121a8b55294dd277bc429cf38e842a30d05
```

**Step 3 — Train and test the CNNs (GPU strongly recommended)**

B0: 2 tasks × 3 architectures × 5 folds = 30 runs.

```bash
for scenario in scenario1 scenario2; do
  for model in resnet18 efficientnet_b0 mobilenet_v3_large; do
    for fold in 1 2 3 4 5; do
      python experiment1_tissue/core_cnn_b0_a1/run_one_core_cnn_fold_resumable.py \
        --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
        --roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
        --scenario "$scenario" --model "$model" --fold "$fold" \
        --experiment-id B0 --out "$RESULTS_ROOT/core_cv/B0/cnn"
    done
  done
done
```

A1: the same command with the A1 roles, `--experiment-id A1`, `--out "$RESULTS_ROOT/core_cv/A1/cnn"`, and **only folds 2 and 4** (12 runs). Folds 1, 3 and 5 are reused from B0.

**Step 4 — Classical machine-learning baseline (CPU)**

```bash
python experiment1_tissue/core_cnn_b0_a1/run_classical_inner_selected.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --experiment-id B0 --folds 1,2,3,4,5 \
  --out "$RESULTS_ROOT/core_cv/B0/classical"

python experiment1_tissue/core_cnn_b0_a1/run_classical_inner_selected.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --roles "$RESULTS_ROOT/core_a1_roles/inner_outer_roles.csv" \
  --experiment-id A1 --folds 2,4 \
  --out "$RESULTS_ROOT/core_cv/A1/classical"
```

In each fold, two feature types × three classifiers (Extra Trees, Linear SVC, LightGBM) are compared on the validation cores; the best is refitted on all 36 non-test cores and tested once.

**Step 5 — Build the manuscript tissue table**

```bash
python experiment1_tissue/core_cnn_b0_a1/export_core_image_metrics.py \
  --results-root "$RESULTS_ROOT/core_cv" \
  --baseline-roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --variant-roles "$RESULTS_ROOT/core_a1_roles/inner_outer_roles.csv" \
  --out "$RESULTS_ROOT/core_report/manuscript_table" \
  --require-complete
```

Output: `core_image_metrics.csv` / `.json` and `per_class_metrics.csv` — core-level and image-level accuracy and balanced accuracy for B0 and A1, all three CNNs and the classical baseline. Balanced accuracy is the mean recall over all classes, computed from the 45 pooled core predictions (not averaged over folds).

Optional aggregate report (accuracy and macro-F1 only):

```bash
python experiment1_tissue/core_cnn_b0_a1/publish_ccast_summary.py \
  --experiment B0 --results-root "$RESULTS_ROOT/core_cv" \
  --baseline-roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --variant-roles "$RESULTS_ROOT/core_a1_roles/inner_outer_roles.csv" \
  --out "$RESULTS_ROOT/core_report/B0"
```

**Step 6 (optional) — Preprocessing sensitivity check**

[`sensitivity_fullframe/`](sensitivity_fullframe/) repeats the CNN runs of this experiment and of Experiment 2 with the full image field kept (letterbox padding instead of cropping). See its README.

## What computer is needed

| Step | Hardware |
|---|---|
| 1, 2, 4, 5 | Ordinary CPU |
| 3 (and 6) | GPU strongly recommended (one GPU per run; 30 + 12 runs). The scripts also run on a CPU, but very slowly |

## Files in this folder

| File | Role |
|---|---|
| `01_qc_make_manifest.py` | Step 1: quality control and image list |
| `core_cnn_b0_a1/make_core_splits.py` | Step 2: B0 folds |
| `core_cnn_b0_a1/verify_core_contract.py` | Step 2: checks no core is in two roles |
| `core_cnn_b0_a1/make_variant_roles.py` | Step 2: A1 folds |
| `core_cnn_b0_a1/run_one_core_cnn_fold_resumable.py` | Step 3: one CNN run (task × model × fold) |
| `core_cnn_b0_a1/dl_grouped_common.py`, `core_eval.py`, `run_contract.py`, `cnn_checkpoint_state.py` | Helpers used by step 3 |
| `core_cnn_b0_a1/run_classical_inner_selected.py`, `03_train_cancer_searchspace_core_cv.py` | Step 4: classical baseline |
| `core_cnn_b0_a1/export_core_image_metrics.py`, `publish_ccast_summary.py` | Step 5: results |
| `core_cnn_b0_a1/run_one_core_cnn_fold.py` | Older version of the step-3 runner without resume support (the original runs used the `_resumable` version) |
| `core_cnn_b0_a1/test_*.py` | Software tests |
| `sensitivity_fullframe/` | Step 6 |

## Limitations

* The split unit is the tissue core; whether cores come from different patients is not encoded in the data.
* Only two Stage III and two Stage IV cores are available, which limits learning and evaluation of these stages.
