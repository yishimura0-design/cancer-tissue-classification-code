# Coauthor runbook — experiments, input paths, commands, expected outputs

**Scope:** code review / reproduction instructions for the reorganized publication repository. Paths below are illustrative **authorized local paths**. No data or private CCAST directory tree is provided through public GitHub. Commands should be launched from the **repository root**, using an appropriate CPU/CUDA Python environment; see [ENVIRONMENT.md](ENVIRONMENT.md) and the `requirements-*.txt` files.

## 0. Set paths once

```bash
cd /YOUR/LOCAL/cancer-tissue-classification-code
export TISSUE_ROOT="/AUTHORIZED_PRIVATE/TISSUE_SOURCE"
export ANGLE_INPUT_DIR="/AUTHORIZED_PRIVATE/ANGLE_TIFFS"
export BFED_ROOT="/AUTHORIZED_PRIVATE/ROOT_CONTAINING_cancer_combine_new"
export RESULTS_ROOT="/AUTHORIZED_PRIVATE_OUTPUT/cancer_results"
mkdir -p "$RESULTS_ROOT"
```

| Path variable or file | Expected content | Source of truth / caveat |
| --- | --- | --- |
| `$TISSUE_ROOT` | Original tissue microscopy directory scanned by `01_qc_make_manifest.py` | Need the actual authorized data and original QC exclusions; do not assume every TIFF was accepted |
| `$RESULTS_ROOT/tissue_qc/manifest_accepted.csv` | Manifest made from tissue images; labels, `filepath`, `relative_path`, `sample_id_from_folder`, `subgroup_id` | The Core B0/A1 input too; original accepted count 16,200 |
| `$ANGLE_INPUT_DIR` | Authorized source Angle Sweep TIFF directory or collection of directories | Infer HPNE/PANC/MIA class and acquisition subgroup; expected 939 accepted images / 37 groups **only if original QC rules/data match** |
| `$RESULTS_ROOT/angle_qc/manifest_clean.csv` | `path`, `label`, `group_key` and audit columns | Output of `angle_classical/00_clean_make_manifest.py`; both Angle pipelines use it |
| `$BFED_ROOT/cancer_combine_new/` | Matched BF and ED image subtrees for HPNE, PANC, MIA | Script discovers class-labeled BF and ED paths and pairs numeric keys; expected 123 pairs |
| `$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv` | Five-fold train/inner-val/outer-test roles at the core level | Output of `make_core_splits.py` |
| `$RESULTS_ROOT/core_a1_roles/inner_outer_roles.csv` | A1 core-swap fold roles | Output of `make_variant_roles.py`; only folds 2 and 4 change |
| `$RESULTS_ROOT/...` | Private results, predictions, caches and checkpoints | Must stay outside the public Git checkout |

## 1. Tissue Classical ML (Normal/Malignant; six-stage classification)

**Input:** authorized tissue microscopy root. **Main evaluation unit:** 225 tissue subgroups/FOVs. **Expected QC input to both Tissue Classical and Core B0/A1:** 16,200 accepted image frames from 45 cores.

```bash
python tissue_classical/01_qc_make_manifest.py \
  --root "$TISSUE_ROOT" --out "$RESULTS_ROOT/tissue_qc"

python tissue_classical/03_train_cancer_searchspace_grouped.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --scenarios scenario1,scenario2 --features all --models all \
  --folds 5 --seed 42 \
  --out "$RESULTS_ROOT/tissue_classical_cv"
```

The historical paper accounting planned 96 configurations, completed 94 and skipped two infeasible LinearSVC/raw-pixel configurations. The specialized `04_merge_all8_selected6_skip_aware.py`, `05_regenerate_selected_best_oof_v6.py`, and `export_paper_classical_ml.py` consume **archival intermediate directory layouts and results**, not only the fresh output directory above. Read `tissue_classical/README.md` and check those scripts' expected root layout before using them; do not present them as turnkey standalone commands. The per-sample prediction exports are not public.

## 2. Core CNN and Core Classical ML: B0 and sensitivity variant A1

**Input:** tissue QC manifest, frozen 45-core fold roles. **Prediction unit:** biological core. B0 is the baseline; A1 changes outer-fold ownership for two cores in folds 2 and 4 and reuses B0 folds 1/3/5. It is not a new cohort.

```bash
python core_cnn_b0_a1/make_core_splits.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --out "$RESULTS_ROOT/core_b0_roles"

python core_cnn_b0_a1/verify_core_contract.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --out "$RESULTS_ROOT/core_roles_audit.json"

python core_cnn_b0_a1/make_variant_roles.py \
  --baseline-roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --out "$RESULTS_ROOT/core_a1_roles"
```

The A1 code asserts the **predeclared C2/C3 swap**; it will reject other fold maps. Keep the previously frozen original role files if matching published data. To run the **B0 CNN campaign** (two scenarios × three ImageNet-pretrained architectures × five folds = 30 model-fold jobs):

```bash
for scenario in scenario1 scenario2; do
  for model in resnet18 efficientnet_b0 mobilenet_v3_large; do
    for fold in 1 2 3 4 5; do
      python core_cnn_b0_a1/run_one_core_cnn_fold_resumable.py \
        --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
        --roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
        --scenario "$scenario" --model "$model" --fold "$fold" \
        --experiment-id B0 --out "$RESULTS_ROOT/core_cv/B0/cnn"
    done
  done
done
```

For **A1 CNN**, use the **A1 roles**, `--experiment-id A1`, a distinct `--out "$RESULTS_ROOT/core_cv/A1/cnn"`, and **only folds 2 and 4** (12 model-fold jobs); reuse B0 outputs for the unchanged folds when making the final pooled comparison. GPU resources and matching pretrained weights are required for the real campaign.

The corresponding **Core Classical ML inner-selected** runner (non-epoch, core-level model selection) is:

```bash
python core_cnn_b0_a1/run_classical_inner_selected.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --experiment-id B0 --folds 1,2,3,4,5 \
  --out "$RESULTS_ROOT/core_cv/B0/classical"

python core_cnn_b0_a1/run_classical_inner_selected.py \
  --manifest "$RESULTS_ROOT/tissue_qc/manifest_accepted.csv" \
  --roles "$RESULTS_ROOT/core_a1_roles/inner_outer_roles.csv" \
  --experiment-id A1 --folds 2,4 \
  --out "$RESULTS_ROOT/core_cv/A1/classical"
```

`03_train_cancer_searchspace_core_cv.py` is also included for exploratory feature/model sweeps. **Do not use outer-fold hyperparameter ranking as independent held-out model selection.** `publish_ccast_summary.py` builds aggregate summaries from stored Core results and role tables:

```bash
python core_cnn_b0_a1/publish_ccast_summary.py \
  --experiment B0 --results-root "$RESULTS_ROOT/core_cv" \
  --baseline-roles "$RESULTS_ROOT/core_b0_roles/inner_outer_roles.csv" \
  --variant-roles "$RESULTS_ROOT/core_a1_roles/inner_outer_roles.csv" \
  --out "$RESULTS_ROOT/core_report/B0"
```

This report is meaningful only after required run artifacts exist in the expected layout.

## 3. Angle Sweep Classical ML

**Input:** authorized Angle TIFFs. **Task:** HPNE/PANC/MIA classification of 939 images, grouped as 37 angle-acquisition subgroups. All images from one subgroup must stay in the same CV fold.

```bash
python angle_classical/00_clean_make_manifest.py \
  --input-dirs "$ANGLE_INPUT_DIR" \
  --out "$RESULTS_ROOT/angle_qc"

python angle_classical/01_build_feature_cache.py \
  --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
  --search-spaces angle_classical/search_spaces.json \
  --out "$RESULTS_ROOT/angle_features"

python angle_classical/02_stage1_select_searchspace_5fold.py \
  --features-dir "$RESULTS_ROOT/angle_features" \
  --models ExtraTrees,RandomForest,LinearSVC,SGD_LogLoss,DecisionTree,LightGBM \
  --n-splits 5 --seed 42 --use-group-weights \
  --out "$RESULTS_ROOT/angle_classical_5fold"
```

Use **`--models all`** to mirror the historical broader 12-classifier exploratory search (10 feature sets × 12 classifiers = 120 configurations × 5 folds = 600 rows). The six-model manuscript subset has 60 configurations × 5 folds = 300 fold rows. Output includes `fold_results.csv`, `all_results.csv`, `ranked_results.csv`, and `stage1_best_setting.json`. Original July CCAST PBS stdout exists in the **private coauthor execution-logs ZIP**, not in public GitHub. This is model-comparison CV, not external held-out testing.

## 4. Angle Sweep CNN

**Input:** the frozen Angle `manifest_clean.csv` above. **Task:** HPNE/PANC/MIA, 3 CNN architectures × 5 outer held-out subgroup folds (15 jobs).

```bash
python angle_cnn/angle_cnn_heldout_cv.py prepare \
  --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
  --out "$RESULTS_ROOT/angle_cnn_heldout"

for model in resnet18 efficientnet_b0 mobilenet_v3_large; do
  for fold in 1 2 3 4 5; do
    python angle_cnn/angle_cnn_heldout_cv.py run \
      --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
      --out "$RESULTS_ROOT/angle_cnn_heldout" --model "$model" --fold "$fold"
  done
done

python angle_cnn/angle_cnn_heldout_cv.py summarize \
  --manifest "$RESULTS_ROOT/angle_qc/manifest_clean.csv" \
  --out "$RESULTS_ROOT/angle_cnn_heldout"
```

Unlike Angle Classical's historical search CV, this design uses an inner validation split plus a held-out outer test. Verify a GPU is present for training and preserve the original split plan.

## 5. BF, ED and BF+ED Classical ML

**Input:** `$BFED_ROOT/cancer_combine_new/`, containing class-specific bright-field and ED image subfolders with matching class/numeric IDs. **123 matched pairs**: HPNE 40, PANC 40, MIA 43. **BFplusED concatenates feature vectors**; it does *not* build a merged PNG.

```bash
export CANCER_DATA_ROOT="$BFED_ROOT"
for stride in 8 16 32; do
  python bf_ed_classical/bfed_rawpix_3fold_cv_testsize.py \
    --cv 5 --cv-type group --eval-mode kfold --seed 42 \
    --stride "$stride" --ed-norm none \
    --modes BFplusED,BF,ED \
    --models ExtraTrees,RandomForest,LinearSVC,SGD_LogLoss,DecisionTree,LightGBM \
    --no-final-model --out-dir "$RESULTS_ROOT/bfed/stride_$stride"
done
```

**Never omit `--cv 5`: the filename/default still refer to three folds.** This command reproduces the six manuscript classifiers (3 modes × 3 strides × 6 models × 5 folds = **270 fold evaluations**), provided that the same 123 pairs, preprocessing and versions are used. The historical source PBS script used `--models all` (a larger exploratory search), and saved `meta.json` confirms `cv=5`, `eval_mode=kfold`, `cv_type=group`, `seed=42`. Relevant outputs: `meta.json`, `cv_fold_results_long.csv`, `cv_per_class_results_long.csv`, and `TABLE2_cv_summary_by_mode_model.csv`. Original July PBS stdout/stderr for all three strides is preserved privately.

## 6. Full-frame CNN padding repeat

**Input:** same protected Core and Angle data and existing frozen fold roles. Preserve the complete image field using 224×224 aspect-ratio-preserving bicubic letterbox/neutral-gray padding instead of center-cropping. **57 training jobs**: 30 B0, 12 A1 and 15 Angle. This is a preprocessing sensitivity study, not a new independent cohort.

Start with **data-free generator checks** from the repository root:

```bash
python fullframe_padding/prepare_fullframe_repeat.py \
  --repo . --output "$RESULTS_ROOT/fullframe_source_code"

python fullframe_padding/build_cached_campaign.py \
  --repo . --output "$RESULTS_ROOT/fullframe_cached_code"
```

These produce source/job definitions **only**, not 57 completed network trainings. Real TIFF conversion/cached image preparation uses `prepare_shared_cache.py --root` pointing to a **private project root** with `qc_outputs_clean/manifest_accepted.csv`, `cancer_angle_professional_pipeline/results_angle_cpu/manifest_clean.csv`, and matching frozen splits. Consult [`fullframe_padding/README.md`](../fullframe_padding/README.md) and [`preflight_fullframe_repeat.py`](../fullframe_padding/preflight_fullframe_repeat.py); PBS templates are CCAST-specific and must be adapted before submission.

## 7. Result files, QA, and scientific interpretation

The private coauthor delivery is split intentionally:

1. `Cancer_Training_Logs_Coauthor_Review.zip` — 114 CNN epoch histories, Core Classical metrics and **newer one-fold (Fold 4) Classical QA reports**.
2. `Cancer_Classical_Original_5Fold_Results.zip` — original July Angle and BF/ED Classical **full five-fold per-configuration metrics** and run metadata.
3. `Cancer_Classical_Original_5Fold_Execution_Logs_Coauthor.zip` — **original July PBS stdout/stderr** for Angle Classical and BF/ED Classical, plus job scripts; paths redacted.

No individual images, source patient IDs, feature caches, or weights are in these publication materials. The two classical Fold 4 checks passed within documented tolerance; they do **not** verify the entire five-fold rerun. See [`IMPLEMENTATION_REVIEW.md`](IMPLEMENTATION_REVIEW.md) and [`REPRODUCIBILITY_QA.md`](REPRODUCIBILITY_QA.md).

**Cautions:** same CV folds may be used both for configuration ranking and reporting, which can yield selection optimism. Filename-derived groups are not proof of patient independence or clinical validity. Numerical equality also depends on the private original dataset, group IDs, software versions and fixed seeds.
