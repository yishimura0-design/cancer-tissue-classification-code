# Guide for coauthors, thesis defense, and reviewers

This document is a **map from manuscript experiment to code**, not a claim that the protected source images can be downloaded.

| Research question | Code folder | Input | Biological grouping and CV | Primary readout | Important caveat |
| --- | --- | --- | --- | --- | --- |
| Normal vs malignant; six-class tissue staging, **classical ML** | `tissue_classical/` | tissue microscopy images | 225 FOV/tissue subgroups, grouped 5 folds | Subgroup macro-F1 and per-class metrics | Do not call these 225 independent cores |
| Normal vs malignant; six-class tissue staging, **CNN** | `core_cnn_b0_a1/` | 16,200 images from 45 cores | 45 cores, outer 5 folds; A1 is sensitivity variant | CORE accuracy, balanced accuracy and macro-F1 | Different grouping from historical classical study; rare stage support |
| HPNE/PANC/MIA, **angle classical ML** | `angle_classical/` | 939 angular images / 37 subgroups | Grouped 5-fold CV; subgroup images stay together | SUBGROUP macro-F1 | CV used to compare feature/model candidates, not an external test |
| HPNE/PANC/MIA, **angle CNN** | `angle_cnn/` | same 939 angular images | Separate inner validation and outer held-out folds, subgroup-disjoint | GROUP accuracy, balanced accuracy and macro-F1 | Original July CNN validation experiment was superseded by independent outer-test version |
| HPNE/PANC/MIA, **BF/ED classical ML** | `bf_ed_classical/` | 123 matched BF/ED pairs | Grouped 5-fold CV, pairing IDs from filename | Fold accuracy, macro-F1, per-class recall | BF+ED is **raw feature concatenation**, NOT merged image nor two-branch CNN |
| Cropping versus complete-field imaging | `fullframe_padding/` | protected Core and Angle images | Same core/group folds as corresponding CNN baseline | Preprocessing sensitivity | Neutral gray padding at 224×224; no consistent improvement |

## How scores are aggregated

* **Image accuracy**: percentage of individually held-out images correctly classified. Images from one specimen/FOV are *not* independent evidence.
* **Group / subgroup accuracy**: one predicted label for each held-out subgroup after pooling views/scores.
* **Core accuracy**: one predicted label per independent biological core. In the Core CNN campaign, 45 cores rather than 16,200 images form the primary sample unit.
* **Balanced accuracy**: mean of recall for the included classes. If a fold lacks support for a class, inspect its class support rather than presenting an artificial 0% recall.
* **Macro-F1**: unweighted mean of per-class F1 across all classes in the chosen definition. It is distinct from ordinary accuracy and weighted F1.
* **B0 vs A1**: A1 varies the fold allocation of two cores, not an independent cohort or replacement dataset.
* **BF vs ED vs BF+ED**: BF is only BF features, ED only ED features, and BF+ED concatenates BF and ED pixel vectors before a classical model.
* **Padding experiment**: complete input image is bicubic-resized to fit a 224×224 square with gray padding, rather than cropped; all images remain private.

## Why model scores may not match across tables

The **225 subgroup** classical study and the **45 biological core** CNN study use different statistical units. Differences between image-, subgroup-, and core-level accuracy are expected. Cross-validation averages across heterogeneous folds also differ from scoring all pooled held-out predictions once.

**All evaluated models' five-fold results** should be reported where available. When the same folds have been used to rank algorithms, the highest reported CV score can have selection-related optimism; do not call the CV itself an independent external test. No patient-level independence or prospective clinical utility is claimed.

## What a reviewer can reproduce

Without original images: inspect source and static contracts, run synthetic/software checks, and regenerate the **57-job full-frame processing campaign definition**. With authorized original images, frozen identifiers and environment: rerun each experiment and compare to the retained fold metrics. The 1-fold targeted QA results are documented in `REPRODUCIBILITY_QA.md`.

The source code was reorganized for readability and local path configuration. Do not assume the reorganized source yields identical binary checkpoints without matching Python/package versions, images, group definitions, fold roles and random seeds.
