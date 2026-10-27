# Experiment 2 — Pancreatic cell lines from angle-sweep images (HPNE / PANC-1 / MIA PaCa-2)

## What this experiment answers

Can a model tell which of three pancreatic cell lines — **HPNE** (non-malignant), **PANC-1** or **MIA PaCa-2** (malignant) — is shown in polarization angle-sweep microscope images?

## The data

| Item | Value |
|---|---|
| Images | **939** |
| Subgroups (one angle series = one subgroup) | **37**: 12 of 72 images and 25 of 3 images |
| HPNE | 315 images / 13 subgroups |
| PANC-1 | 321 images / 15 subgroups |
| MIA PaCa-2 | 303 images / 9 subgroups |

A **subgroup** is the set of images in the same folder with the same file name apart from the angle. All images of a subgroup always stay together. Subgroups are defined from file names; whether two subgroups come from different dishes or cultures is not encoded in the data.

## How the evaluation works

**CNNs (the results reported in the manuscript)**

* The 37 subgroups are divided into **5 folds**, balanced by cell line (seed 42).
* In each of the 5 rounds, one fold is the **test set**; of the remaining subgroups, **20% are a validation set** used only to decide when to stop training, and the rest are the **training set**.
* Every subgroup is tested **exactly once**, by a model trained from scratch without it.
* A subgroup's prediction is the average of its images' class probabilities. Results are reported at both the subgroup and the image level.

**Classical machine learning (the "previously evaluated" reference models)**

* Grouped five-fold cross-validation over the same 37 subgroups, comparing 10 image-feature types × 6 classifiers.
* The same folds are used to compare the configurations, so this is a model comparison, not an independent test.

## Steps — run from the repository root

```bash
export ANGLE_INPUT_DIR="/AUTHORIZED_PRIVATE/ANGLE_TIFFS"
export RESULTS_ROOT="/AUTHORIZED_PRIVATE_OUTPUT/cancer_results"
mkdir -p "$RESULTS_ROOT"
```

**Step 1 — Build the image list with subgroup keys** (used by both the CNN and the classical models)

```bash
python experiment2_angle_sweep_cells/angle_classical/00_clean_make_manifest.py \
  --input-dirs "$ANGLE_INPUT_DIR" \
  --out "$RESULTS_ROOT/angle_qc"
```

Output: `manifest_clean.csv` with `path`, `label`, `group_key` (expected 939 images, 37 subgroups).

**Step 2 — CNNs (GPU required; 3 architectures × 5 folds = 15 runs)**

```bash
python experiment2_angle_sweep_cells/angle_cnn/angle_cnn_heldout_cv.py prepare \
  --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
  --out "$RESULTS_ROOT/angle_cnn_heldout"

for model in resnet18 efficientnet_b0 mobilenet_v3_large; do
  for fold in 1 2 3 4 5; do
    python experiment2_angle_sweep_cells/angle_cnn/angle_cnn_heldout_cv.py run \
      --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
      --out "$RESULTS_ROOT/angle_cnn_heldout" --model "$model" --fold "$fold"
  done
done

python experiment2_angle_sweep_cells/angle_cnn/angle_cnn_heldout_cv.py summarize \
  --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
  --out "$RESULTS_ROOT/angle_cnn_heldout"
```

Output: `heldout_5fold_summary.json` — pooled subgroup-level and image-level accuracy, balanced accuracy and per-class recall for each architecture (the manuscript angle-sweep table) — and `fold_results.csv`.

**Step 3 — Classical machine learning (CPU)**

```bash
python experiment2_angle_sweep_cells/angle_classical/01_build_feature_cache.py \
  --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
  --search-spaces experiment2_angle_sweep_cells/angle_classical/search_spaces.json \
  --out "$RESULTS_ROOT/angle_features"

python experiment2_angle_sweep_cells/angle_classical/02_stage1_select_searchspace_5fold.py \
  --features-dir "$RESULTS_ROOT/angle_features" \
  --models ExtraTrees,RandomForest,LinearSVC,SGD_LogLoss,DecisionTree,LightGBM \
  --n-splits 5 --seed 42 --use-group-weights \
  --out "$RESULTS_ROOT/angle_classical_5fold"
```

Output: `fold_results.csv`, `all_results.csv`, `ranked_results.csv`, `stage1_best_setting.json`. Use `--models all` to repeat the broader 12-classifier exploratory search of the original runs.

## Preprocessing sensitivity check

The full-image (letterbox padding) repeat of these CNN runs is in [`../experiment1_tissue/sensitivity_fullframe/`](../experiment1_tissue/sensitivity_fullframe/), together with the tissue repeat.

## What computer is needed

| Step | Hardware |
|---|---|
| 1, 3 | Ordinary CPU |
| 2 | GPU (one GPU per run; 15 runs) |

## Files in this folder

| File | Role |
|---|---|
| `angle_classical/00_clean_make_manifest.py` | Step 1: image list and subgroup keys |
| `angle_cnn/angle_cnn_heldout_cv.py` | Step 2: fold plan, one CNN run, pooled summary |
| `angle_cnn/dl_common.py` | Helpers for step 2 (transforms, models, metrics) |
| `angle_classical/01_build_feature_cache.py`, `feature_lib.py`, `search_spaces.json` | Step 3: image features |
| `angle_classical/02_stage1_select_searchspace_5fold.py`, `models_lib.py`, `ml_eval_lib.py` | Step 3: grouped cross-validation |
| `angle_cnn/test_angle_cnn_heldout_cv.py` | Software tests |
