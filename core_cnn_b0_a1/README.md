# Core CNN — B0 / A1, 45 biological cores

**Inputs:** 16,200 private microscope images from 45 cores (5 FOV × 72 views per core). Binary Normal/Malignant and six-class Normal/IB/IIA/IIB/III/IV targets.

**Primary unit:** biological core, not individual image. Outer five folds are disjoint by `sample_id_from_folder`. B0 is the baseline; A1 exchanges specified core fold ownership as a split-sensitivity analysis, not a separate cohort.

## Reproduction workflow (authorized data only)

```bash
python core_cnn_b0_a1/make_core_splits.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --out /PRIVATE_RESULTS/core_roles

python core_cnn_b0_a1/verify_core_contract.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --roles /PRIVATE_RESULTS/core_roles/inner_outer_roles.csv \
  --out /PRIVATE_RESULTS/core_audit.json

python core_cnn_b0_a1/run_one_core_cnn_fold_resumable.py \
  --manifest /PRIVATE/qc/manifest_accepted.csv \
  --roles /PRIVATE_RESULTS/core_roles/inner_outer_roles.csv \
  --scenario scenario1 --model resnet18 --fold 1 \
  --experiment-id B0 --out /PRIVATE_RESULTS/core_cnn
```

Valid models: ResNet18, EfficientNet-B0, MobileNetV3-Large (ImageNet pretrained). B0 comprises 2 tasks × 3 architectures × 5 folds = 30 model-fold jobs; A1 reuses B0 outputs for unchanged folds and retrains only folds 2 and 4 (12 jobs). Validation decides epochs; **outer test does not select checkpoints**. Training code is in this folder, with resumed checkpoints and shared transforms in the adjacent modules.

Manifests must preserve `filepath`, `relative_path`, `sample_id_from_folder`, `subgroup_id`, `scenario1_label`, and `scenario2_label` consistent with the original QC. These columns and any real IDs should not be committed.

**Limitations:** The original split proves core-disjoint test data, not patient independence; certain cancer stages have very few core samples.
