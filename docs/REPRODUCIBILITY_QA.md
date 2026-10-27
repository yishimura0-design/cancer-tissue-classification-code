# Publication code quality and reproducibility evidence

## Previously completed real-data one-fold smoke test (2026-10-20)

The user ran the isolated one-fold QA on authorized private CCast data, comparing selected original Fold 4 metrics with the repeat. No raw data, sample-level predictions, log files, or person-level identifiers are published here.

| Experiment | Fixed model, features | Statistic | Historical | Rerun | Absolute delta |
| --- | --- | --- | ---: | ---: | ---: |
| Angle classical | ExtraTrees / `S09_combined_small_128` / Fold 4 | Image accuracy | 0.4966887417 | 0.4966887417 | 0 |
| Angle classical | same | Subgroup accuracy | 0.6923076923 | 0.6923076923 | 0 |
| Angle classical | same | Subgroup macro-F1 | 0.6363636364 | 0.6363636364 | 0 |
| BF+ED classical | LightGBM / stride 16 / Fold 4 | Accuracy | 0.625000 | 0.625000 | 0 |
| BF+ED classical | same | Macro-F1 | 0.626984 | 0.621078 | 0.005906 |

Both reports explicitly showed `PASS` and `train_test_group_overlap = 0`. The small BF+ED macro-F1 discrepancy is disclosed rather than described as an exact match. The **underlying IDs may not imply biological independence** unless the data custodian validates them.

## Publication preparation checks (2026-10-20)

* Parsed/compiled imported Python sources without the protected images.
* Reorganized experiment directories to match manuscript questions.
* Removed absolute shared-cluster paths from the added source snapshots.
* The full-frame generator produced **57 training job definitions**; the cache-backed campaign builder printed `CACHED_CAMPAIGN_BUILD_PASS` (no GPU training, no image reading).
* Checked that the publication repository does not include original microscopy images, model checkpoints, raw feature matrices or protected manifests.
* Final real-data numerical reproducibility beyond the targeted Fold 4 checks **remains unverified in this publication copy**. Full GPU reruns have not been performed from the reorganized code.

## Follow-up before public release

1. Verify that all five folds use the exact manuscript split / label definitions.
2. Run package unit tests in the documented Python environment, not only CPython `compile`.
3. Confirm permission from authors/custodians to release the **code** and decide its license.
4. Inspect Git history as well as current files for sensitive data before enabling Public access.
5. Tag an immutable release, optionally archive through Zenodo, and update Code Availability wording to refer to that fixed version.

**A passed fold smoke test does not establish statistical unbiasedness, clinical validity, or patient-level independence.**

## Additional evidence recovered on CCAST (2026-10-27)

Original July 2026 execution records were located, independently of the newer Fold 4 verification:

- **Angle Classical:** `angle_stage1_cv.o713717` is the original completed five-fold search stdout. Original `fold_results.csv` has 600 rows across 120 exploratory configurations, all five folds present. Six manuscript classifiers correspond to 300 rows.
- **BF/ED Classical:** `bfed_5fold_stride.pbs` explicitly invokes `--cv 5 --eval-mode kfold --cv-type group --seed 42` for strides 8/16/32; each saved stride `meta.json` confirms it. All 270 fold rows for 54 manuscript configurations exist. Original `bfed_5fold_stride.o724194.0/.1/.2` end with `[DONE] Raw-pixel classification evaluation complete.`
- Sanitized copies of the original stdout/stderr and PBS scripts were placed in the separately shared **private** `Cancer_Classical_Original_5Fold_Execution_Logs_Coauthor.zip`; original five-fold metrics are in `Cancer_Classical_Original_5Fold_Results.zip`. Neither ZIP is hosted in this repository.

A fresh clone of `main` was also checked with the four built-in source-only tests and `compileall`, and both data-free full-frame generators again produced 57 jobs. See [Implementation review](IMPLEMENTATION_REVIEW.md) and [Coauthor runbook](COAUTHOR_RUNBOOK.md). This remains **partial implementation verification**, not a fresh complete five-fold CPU/GPU rerun on real images.


### BF/ED grouped-CV fallback correction

In the 2026-10-27 public code review, an unsafe fallback was found: missing `StratifiedGroupKFold` previously caused `--cv-type group` to use ungrouped `StratifiedKFold`. The code now **fails closed** instead of risking group leakage. This affects only unsupported/older scikit-learn environments; it does not alter the verified original CCAST 5-fold results obtained with scikit-learn 1.6.1. No full numeric reproduction is implied.
