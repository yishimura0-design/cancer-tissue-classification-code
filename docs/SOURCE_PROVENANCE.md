# Source provenance

* **BF/ED** and **angle classical**: source scripts and search-space configuration were recovered from the original research's private Google Drive implementation-review bundles, dated July 2026. The original BF/ED raw-pixel source name includes `3fold`, but the **published target protocol must explicitly run grouped `--cv 5`**, as in its retained execution records.
* **Tissue classical**: the primary grouped 5-fold trainer, QC script, 54/96-configuration result-merger logic, and selected-best OOF v6 were recovered from the original Drive review package.
* **Core CNN and independent Angle CNN**: source imported from the maintained private `yishimura0-design/cancer` repository, branch `main` (snapshot available in Git history, version at publication export time).
* **Full-frame padding**: source imported from `yishimura0-design/cancer`, branch `experiment/fullframe-cnn-repeat-v1`; original 57-job build contract preserved.

## Publication-only portability edits

* Replace BF/ED hard-coded input directories with `CANCER_DATA_ROOT`, without modifying its pairing, stride, model, CV or evaluation logic.
* Angle classical image discovery requires `--input-dirs`; old inaccessible CCast paths were removed from defaults. Fold logic retained.
* Angle CNN defaults no longer point to private CCast directories. Always pass `--manifest` / `--out` explicitly.
* Reorganize all Core/Angle/full-frame modules into per-experiment folders. Full-frame generator dependencies now point to those subfolders; the 57-job build was verified.
* Former cluster-root defaults in supporting scripts are replaced with `./private_data` or `CANCER_DATA_ROOT`. Site-specific PBS scripts, if used, require local adaptation and are not proof of execution.

The original projects, original published numeric results, and the private images were never overwritten by publication packaging. Review this record and the per-experiment README files before attempting to reproduce.
