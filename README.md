# Cancer Tissue Classification — Publication Code

**Research software accompanying an experimental microscopy classification manuscript.**

> **Release status: PREPARATION / PRIVATE.** This repository contains source code and documentation only. It does **not** contain the underlying microscopy images, feature caches, individual-sample manifests, model checkpoints, or credentials. The source archive is being reconciled with the paper's final tables. Do not cite this repository as a finalized public release yet.

## Find the code for each experiment

| Paper experiment | Directory | Input | Evaluation unit |
| --- | --- | --- | --- |
| Tissue classical ML (Scenario 1 / Scenario 2) | [tissue_classical/](tissue_classical/) | Tissue microscopy images | 225 tissue subgroups; grouped 5-fold CV |
| BF / ED / BF+ED classical ML | [bf_ed_classical/](bf_ed_classical/) | Matched bright-field and ED images | 123 paired IDs; grouped 5-fold CV |
| Angle Sweep classical ML | [angle_classical/](angle_classical/) | 939 angular images | 37 subgroups; grouped 5-fold CV |
| Core CNN B0 / A1 | [core_cnn_b0_a1/](core_cnn_b0_a1/) | 16,200 images across 45 biological cores | 45 independent cores; outer 5-fold CV |
| Angle Sweep CNN | [angle_cnn/](angle_cnn/) | 939 angular images | 37 subgroups; nested train/validation and independent outer 5-fold CV |
| Full-frame padding sensitivity repeat | [fullframe_padding/](fullframe_padding/) | Same source images as Core / Angle CNN | Same held-out biological grouping, full-frame letterbox resize |

Do not conflate **45 biological cores** (new Core CNN experiment) with **225 tissue subgroups/FOVs** (historical tissue classical ML experiment). Neither constitutes patient-independent external validation.

## Data availability and permissions

**No dataset is released here.** Images are controlled by their owners, not the repository maintainer. Do not add TIFF, PNG, JPG, raw datasets, per-image filenames, image-level predictions, protected sample metadata, cached features, or model checkpoints to Git.

A person authorized to use the original dataset must provide the images **locally** and configure their own paths. A synthetic test fixture, if present, validates software behavior and does not reproduce the reported biological results.

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the publication statement.

## Reproducibility and evaluation

* Three target classes for BF/ED and Angle: `HPNE`, `PANC`, `MIA`.
* Binary Core task: Normal vs Malignant. Six-class Core task: Normal, Stage IB, IIA, IIB, III, IV.
* Train/validation/test isolation is at the **specified grouping unit**. A filename-derived ID cannot independently establish biological independence.
* Report model-by-model **all five folds** and group-level metrics. Imaging views are correlated, so frame-level performance is secondary.
* The original classical ML code uses **no validation-driven checkpoint early stopping**. Classical CV scores were used to compare candidate models; a separate external cohort is not claimed.
* **Verified targeted one-fold repeat (2026-10-20):** Angle classical ExtraTrees + S09, Fold 4: image accuracy 0.4966887417, subgroup accuracy 0.6923076923, subgroup macro-F1 0.6363636364 (all exactly matched historical). BF+ED LightGBM, stride 16, Fold 4: accuracy 0.625 (exact), macro-F1 0.621078 vs historical 0.626984 (delta 0.005906). Both reported zero train/test grouping-key overlap. This is a **smoke check, not full five-fold reproducibility or proof of biological independence**.
* Full-frame padded CNN repeat had 57 completed model/fold jobs. It is an exploratory preprocessing sensitivity check, not a new independent cohort.

## How to reproduce without public data

1. Obtain permission from the actual image/data owner and prepare local source images.
2. Use experiment-specific manifests and configuration described in each subfolder. Any paths in old HPC scripts are **examples**, not a publicly accessible dataset.
3. Create the Python environment, check pinned/runtime versions in the experiment README and environment notes, and run syntax/group split/metric tests before training.
4. Run a single fold first and confirm it reproduces the retained paper metrics; only then consider a full run.
5. Save outputs **outside this git repository**. Never commit samples, image names, identifying metadata, fitted estimators, or intermediate feature arrays.

## Citation and licensing

The publication version, authorship, code license and final DOI/tag must be approved before making the repository public. No blanket open-source license is granted while permissions are under review. Please cite the paper and fixed release/commit after approval.

## Code availability for the manuscript

Use the draft in [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md). Replace the pending-release statement with the public DOI/tag only when the release is approved and publicly accessible.

## Start here for a paper or thesis defense

- [Experiment-to-code map and explanations](docs/PAPER_EXPERIMENT_MAP.md)
- [Reproducibility test evidence](docs/REPRODUCIBILITY_QA.md)
- [Recorded scientific software versions](docs/ENVIRONMENT.md)
- [Original code provenance and portability changes](docs/SOURCE_PROVENANCE.md)
- [Classical ML pinned dependencies](requirements-classical.txt)
- [CNN dependency guidance](requirements-cnn.txt)

Run source-only tests: `python -m unittest discover -s tests -v`. No image downloads are required for these packaging tests.

