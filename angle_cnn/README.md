# Angle Sweep CNN — independent outer group test

**Task:** HPNE / PANC / MIA, 939 angular images in 37 subgroups. Models: ResNet18, EfficientNet-B0, MobileNetV3-Large with 224×224 ImageNet-normalized inputs.

Use `angle_cnn_heldout_cv.py` and `dl_common.py`, not the historical Stage 1 CNN source that selected and reported performance on the same fold. This updated version separates **inner training/validation** from **outer held-out test**, keeping all angles from each subgroup in one role.

## Run with a locally authorized manifest

The manifest CSV must identify image path, class label and subgroup key. Accepted labels: `HPNE`, `PANC`, `MIA`.

```bash
python angle_cnn/angle_cnn_heldout_cv.py prepare \
  --manifest /PRIVATE/angle_manifest_clean.csv \
  --out /PRIVATE_RESULTS/angle_cnn

python angle_cnn/angle_cnn_heldout_cv.py run \
  --manifest /PRIVATE/angle_manifest_clean.csv \
  --out /PRIVATE_RESULTS/angle_cnn \
  --model efficientnet_b0 --fold 1

python angle_cnn/angle_cnn_heldout_cv.py summarize \
  --manifest /PRIVATE/angle_manifest_clean.csv \
  --out /PRIVATE_RESULTS/angle_cnn
```

The original research runs used all three models × 5 folds (15 jobs); report GROUP metrics with denominator 37 rather than treating the 939 images as independent samples. Small MIA group count (9) limits precision. No original images or individual prediction exports are present in the repository.
