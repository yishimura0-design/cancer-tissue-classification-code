# Implementation cross-check — coauthor handoff (2026-10-27)

This is a **scoped verification report**, not a certification of end-to-end scientific correctness. Checks were made against the cloned current GitHub `main` and retained private CCAST result files/scripts, without copying data into GitHub or retraining models.

## Checks performed on the publication code

| Check | Outcome | Scope |
| --- | --- | --- |
| Repository source tree | PASS | Experiment directories include Tissue Classical; Core B0/A1 CNN and Classical; Angle Classical; Angle CNN; BF/ED Classical; full-frame |
| `python3 -m unittest discover -s tests -v` | PASS (4 tests) | Source parse; schema examples; disallowed data types; simple private-path scan |
| Python compilation | PASS | `compileall -q` over all six experiment directories |
| Python CLI checks | See recorded CLI validation below | `--help` paths exercise imports/argparse only; no datasets or training |
| Full-frame generator | PASS | `prepare_fullframe_repeat.py` generated 57 job definitions in a temporary directory |
| Full-frame cached generator | PASS | `build_cached_campaign.py` printed `CACHED_CAMPAIGN_BUILD_PASS: 57 jobs` |
| Source-bundle completeness | PARTIAL | Main experimentation scripts and required adjacent modules exist. Dataset/prespecified splits/archival PBS environment aren't embedded |
| End-to-end five-fold real-data rerun using reorganized publication source | **NOT PERFORMED** | Original protected dataset; large CPU/GPU campaigns and exact environment required |
| Patient-independent biological validation | **NOT ESTABLISHED** | Source group identifiers and split checks do not prove patient independence |

## Original experiment evidence read on CCAST

**Angle Classical:**
- Original `pbs_stage1_5fold_search.pbs` sets `--n-splits 5 --seed 42`, model search space `all`, and subgroup weights.
- The historical `fold_results.csv` contains **600 fold rows**: 10 feature representations × 12 exploratory classifiers × all five folds, without missing folds.
- Original PBS stdout `angle_stage1_cv.o713717` exists and contains the completed Stage 1 search plus final best-setting record.
- Manuscript focuses on six of the classifiers (300 fold rows).

**BF/ED Classical:**
- Original `bfed_5fold_stride.pbs` explicitly invokes `--cv 5 --eval-mode kfold --cv-type group --seed 42` and strides 8/16/32.
- Original `meta.json` in each stride directory independently confirms these settings.
- Manuscript six-classifier subset: **54 configurations × five folds = 270 complete fold rows**; original broader search contains additional exploratory models.
- Original array-job stdout `bfed_5fold_stride.o724194.{0,1,2}` for three strides each ends with `[DONE] Raw-pixel classification evaluation complete.` Each original stderr file is retained privately as well.

**Targeted new Fold 4 QA (2026-10-20):**
- Angle Classical ExtraTrees/S09: image accuracy, subgroup accuracy and subgroup macro-F1 matched original Fold 4 exactly.
- BF+ED LightGBM/stride 16: accuracy matched; macro-F1 difference was **0.005906**, within the stated QA tolerance.
- Both reported zero **group-key** train/test overlap. These are **one-fold smoke checks**, not a five-fold reproduction or proof of biological independence.

**CNN/Full-frame:** Original per-fold histories exist for Core B0 (30), A1 (12), Angle CNN (15), Full-frame (57). The original July/October campaign evidence is distinct from the source-only generator checks above.

## Items for coauthor validation before manuscript use

1. Confirm the **exact authorized original dataset**, QC exclusions and manuscript denominators (Tissue 16,200/225 FOV; Core 16,200/45 cores; Angle 939/37 groups; BF/ED 123 pairs).
2. Verify **frozen original group identifiers / five-fold assignments**, especially distinguishing patient, core, subgroup and paired numeric key.
3. Ensure **CNN train/inner-validation/outer-test** separation, early-stopping policy and any A1 B0-fold reuse match the reported manuscript tables. Do not mix full-frame repeats with baseline.
4. Review all **reported six-classifier subsets versus broader exploratory 12-model searches** for Angle and BF/ED; avoid presenting fold-based model selection as independent validation.
5. Repeat the remaining folds under the **documented Python and image-preprocessing environment** if full numerical reproducibility is required; do not describe the one-fold checks as full confirmation.
6. Confirm authorship/permission, code license, and public-GitHub visibility before citing the repository as a finalized release.

## Code review outcome

The source files compile and core packaging checks pass; original execution evidence supports the written **five-fold Classical** protocol. The checks do **not** establish that every executable branch is error-free, nor do they replace a real-data complete run. Any numerical, biological, or author-facing claim should be checked against the private original results and fixed dataset/split versions.


## Safety correction to BF/ED code (2026-10-27)

During review, we found that when `StratifiedGroupKFold` was unavailable, `make_cv_splits(..., cv_type="group", eval_mode="kfold")` fell back to ungrouped `StratifiedKFold`. That is unsafe for grouped data because it can split repeated pair keys across train/test. **Fixed:** the public source now raises a clear `RuntimeError` instead of silently switching split types. The historical environment was scikit-learn 1.6.1, where `StratifiedGroupKFold` exists; this fix does **not** change the historical five-fold protocol or saved results. Full real-data reruns remain unperformed.

The original broader model search also includes more exploratory classifiers than the six selected for manuscript figures. The reviewed runbook explicitly uses the six-model subset and `--cv 5` to prevent accidental use of the three-fold script default.
