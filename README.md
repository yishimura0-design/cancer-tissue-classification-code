# Cancer Tissue Classification — Coauthor Reproduction Guide

**Status: public code-review snapshot, not a finalized or approved research-software release.** The repository is currently visible on GitHub. It contains implementation source and documentation, **not the microscopy datasets**. Coauthors should confirm methods, input grouping, split definitions, actual run settings, and numerical results before handing code/results to the manuscript author. Code availability does not imply permission to publish the protected images.

**First read:** [Experiment-by-experiment input paths and exact commands](docs/COAUTHOR_RUNBOOK.md) · [Implementation checks and open items](docs/IMPLEMENTATION_REVIEW.md) · [Experiment-to-paper map](docs/PAPER_EXPERIMENT_MAP.md).

## Experiments and implemented source

| Paper experiment | Code directory / primary entrypoint | Required private input | Evaluation |
| --- | --- | --- | --- |
| Tissue Classical ML | [`tissue_classical/`](tissue_classical/) — `01_qc_make_manifest.py`, `03_train_cancer_searchspace_grouped.py` | Authorized tissue image root (16,200 accepted images after QC) | 225 FOV/tissue subgroups, grouped five-fold CV; binary and six-class tasks |
| Core CNN B0/A1 + Core Classical | [`core_cnn_b0_a1/`](core_cnn_b0_a1/) — `make_core_splits.py`, `run_one_core_cnn_fold_resumable.py`, `run_classical_inner_selected.py` | Tissue QC manifest, core role tables, 16,200 authorized images | 45 physical cores, five outer folds; A1 retrains changed folds 2/4 |
| Angle Classical ML | [`angle_classical/`](angle_classical/) — `00_clean_make_manifest.py`, `01_build_feature_cache.py`, `02_stage1_select_searchspace_5fold.py` | Authorized angle TIFF folders | 939 images / 37 filename-derived subgroups, grouped five-fold CV |
| Angle CNN | [`angle_cnn/`](angle_cnn/) — `angle_cnn_heldout_cv.py` | Same authorized angle manifest | 3 CNNs × 5 outer held-out subgroup folds |
| BF / ED / BF+ED Classical ML | [`bf_ed_classical/`](bf_ed_classical/) — `bfed_rawpix_3fold_cv_testsize.py` | Paired BF and ED TIFF folders, 123 matched pairs | `--cv 5 --cv-type group --eval-mode kfold`; 3 modes × 3 strides × 6 manuscript classifiers |
| Full-frame letterbox CNN repeat | [`fullframe_padding/`](fullframe_padding/) — `prepare_fullframe_repeat.py`, `build_cached_campaign.py` | Authorized original Core/Angle images and frozen fold roles | Same fold structure, 57 jobs (30 B0 + 12 A1 + 15 Angle) |

**Important:** BF+ED is **concatenated raw-pixel feature vectors**, not a merged image or dual-branch CNN. The filename `bfed_rawpix_3fold_cv_testsize.py` predates the published five-fold configuration; **running the script without `--cv 5` uses its three-fold default and does not reproduce the paper**.

The historical **Tissue Classical** grouping (225 FOV/tissue subgroups) is distinct from the **Core CNN/Core Classical** grouping (45 physical cores). Neither should be called patient-independent validation.

## Running code: start here

Use a terminal at the repository root on a machine with permitted access to the source data. These are **input roles**, not public download locations:

```bash
export TISSUE_ROOT="/AUTHORIZED_PRIVATE/TISSUE_SOURCE"
export ANGLE_INPUT_DIR="/AUTHORIZED_PRIVATE/ANGLE_TIFFS"
export BFED_ROOT="/AUTHORIZED_PRIVATE/ROOT_CONTAINING_cancer_combine_new"
export RESULTS_ROOT="/AUTHORIZED_PRIVATE_OUTPUT/cancer_results"
mkdir -p "$RESULTS_ROOT"
```

Follow [`docs/COAUTHOR_RUNBOOK.md`](docs/COAUTHOR_RUNBOOK.md) for complete, copyable per-experiment commands, all five folds and associated results, an input/output path table, and caveats about which auxiliary pipelines still require the original private folder layout. **Never put data, cache files, original execution logs, or predictions inside the Git working tree.**

## Original results versus newer verification

- **Original Angle Classical:** the retained July 2026 results contain **120 configurations × 5 folds = 600 fold rows** (10 feature spaces × 12 exploratory models). The paper focuses on six classifiers (60 configurations, 300 fold rows).
- **Original BF/ED Classical:** the retained July 2026 PBS job script explicitly uses `--cv 5 --cv-type group --eval-mode kfold --seed 42`. All **54 paper configurations × 5 folds = 270 fold rows** are present across strides 8, 16 and 32.
- **2026-10-20 targeted reproducibility QA:** one selected **Fold 4** per Classical experiment, compared against the original Fold 4. These are **not** five-fold reruns. See [QA documentation](docs/REPRODUCIBILITY_QA.md).
- **CNN training records:** the private coauthor review ZIP contains 114 CNN per-fold epoch histories: Core B0 30, Core A1 12, Angle 15 and full-frame repeat 57. The private ZIP is separate from this code repository.

## Cross-check status

- Source-only tests and Python compilation passed on the cloned review snapshot, and the data-free full-frame generators produced the expected 57 job definitions.
- Original Classical five-fold records and PBS scripts were cross-checked against the documented protocols.
- **Not yet verified:** full five-fold model-by-model numerical reproduction from this reorganized repository with the protected dataset, all GPU training runs, environmental parity, and independent biological/patient grouping claims. See [`docs/IMPLEMENTATION_REVIEW.md`](docs/IMPLEMENTATION_REVIEW.md).

## Permissions, safety, and release status

The original microscopy TIFFs, private manifest rows/identifiers, feature caches, per-image predictions, model weights, and unsanitized PBS logs are deliberately excluded. Access to protected inputs and coauthor ZIPs must be arranged privately with the data custodian. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md), [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md), [software environment](docs/ENVIRONMENT.md) and [source provenance](docs/SOURCE_PROVENANCE.md).

The repository is **currently public**, but the publication version, authorship, license, release tag and permitted content still need coauthor/custodian approval. Do not describe it as an approved final research release or claim clinical validation.