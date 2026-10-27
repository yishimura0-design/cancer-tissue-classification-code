# Cancer Tissue Classification — Coauthor Reproduction Guide

**Status: public code-review snapshot, not a finalized or approved research-software release.** The repository is currently visible on GitHub. It contains implementation source and documentation, **not the microscopy datasets**. Coauthors should confirm methods, input grouping, split definitions, actual run settings, and numerical results before handing code/results to the manuscript author. Code availability does not imply permission to publish the protected images.

## Data availability

The microscopy images analysed in this study (pancreatic tissue microarray images and pancreatic cell-line angle-sweep, brightfield and elliptical dichroism images) are not publicly available and are not included in this repository. Only the code implementation is released; the repository contains no images, patient or sample identifiers, per-image predictions, or trained model weights.

## Organized by manuscript experiment

Each experiment folder has its own README with the question it answers, the data it needs, numbered run steps, which manuscript result each output gives, and the hardware required.

| Experiment | Question | Folder | Models |
| --- | --- | --- | --- |
| **1. Pancreatic tissue** | Normal vs malignant; normal vs five cancer stages | [`experiment1_tissue/`](experiment1_tissue/) | 3 CNNs + classical baseline; 45 cores, five folds; B0 primary, A1 sensitivity |
| **2. Cell lines, angle sweep** | HPNE vs PANC-1 vs MIA PaCa-2 | [`experiment2_angle_sweep_cells/`](experiment2_angle_sweep_cells/) | 3 CNNs (reported) + classical reference models; 939 images, 37 subgroups |
| **3. Cell lines, BF / ED** | Does ED add information to brightfield? | [`experiment3_bf_ed_cells/`](experiment3_bf_ed_cells/) | 6 classical classifiers; 123 matched pairs; BF, ED, BF+ED |

```
cancer-tissue-classification-code/
├── experiment1_tissue/
│   ├── README.md                    ← start here for Experiment 1
│   ├── 01_qc_make_manifest.py       ← step 1: image quality control
│   ├── core_cnn_b0_a1/              ← folds, CNN and classical training, results table
│   └── sensitivity_fullframe/       ← optional preprocessing check (Experiments 1 and 2)
├── experiment2_angle_sweep_cells/
│   ├── README.md                    ← start here for Experiment 2
│   ├── angle_cnn/                   ← CNNs (manuscript results)
│   └── angle_classical/             ← image list + classical reference models
├── experiment3_bf_ed_cells/
│   ├── README.md                    ← start here for Experiment 3
│   └── bf_ed_classical/
├── docs/                            ← software environment (versions used)
├── examples/                        ← example image tables (placeholder rows, no data)
└── tests/                           ← software checks (no data needed)
```

**Important:**

* All reported tissue results use **45 physical tissue cores** as the split unit; all images of a core stay in one fold. Neither the core nor the subgroup grouping should be called patient-independent validation.
* BF+ED is **concatenated raw-pixel feature vectors**, not a merged image or dual-branch CNN. The filename `bfed_rawpix_3fold_cv_testsize.py` predates the published five-fold configuration; **running the script without `--cv 5` uses its three-fold default and does not reproduce the paper**.

## Running code

Use a terminal at the repository root on a machine with permitted access to the source data. These are **input roles**, not public download locations:

```bash
export TISSUE_ROOT="/AUTHORIZED_PRIVATE/TISSUE_SOURCE"
export ANGLE_INPUT_DIR="/AUTHORIZED_PRIVATE/ANGLE_TIFFS"
export BFED_ROOT="/AUTHORIZED_PRIVATE/ROOT_CONTAINING_cancer_combine_new"
export RESULTS_ROOT="/AUTHORIZED_PRIVATE_OUTPUT/cancer_results"
mkdir -p "$RESULTS_ROOT"
```

Then follow the README of the experiment you want to run. **Never put data, cache files, original execution logs, or predictions inside the Git working tree.**

Software checks that need no data: `python -m unittest discover -s tests -v`.

## How results are computed

* **Image accuracy**: share of individually held-out images classified correctly. Images from the same field of view or subgroup are correlated, so they are not independent evidence.
* **Subgroup accuracy** (Experiment 2): one prediction per held-out subgroup, from the average of its images' class probabilities.
* **Core accuracy** (Experiment 1): one prediction per held-out tissue core, from the average over its 72 angles and then its 5 fields of view. The 45 cores, not the 16,200 images, are the primary unit.
* **Balanced accuracy**: mean recall over all classes.
* **Macro-F1**: unweighted mean of the per-class F1 scores.
* Tissue and angle-sweep results are computed from the pooled held-out predictions of all five folds; BF/ED results are reported as mean ± standard deviation across the five folds.
* **B0 vs A1**: A1 changes the fold allocation of two cores; it is a sensitivity analysis on the same specimens, not an independent cohort.
* **BF vs ED vs BF+ED**: BF uses only brightfield pixels, ED only ED pixels, and BF+ED joins the two pixel vectors before a classical model.

Software versions are listed in [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Permissions, safety, and release status

The original microscopy TIFFs, private manifest rows/identifiers, feature caches, per-image predictions, model weights, and unsanitized PBS logs are deliberately excluded.

The repository is **currently public**, but the publication version, authorship, license, release tag and permitted content still need coauthor/custodian approval. Do not describe it as an approved final research release or claim clinical validation.
