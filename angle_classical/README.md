# Angle Sweep classical ML — grouped five-fold model comparison

**Target:** HPNE/PANC/MIA from angular microscope images. Original design: 939 images in 37 filename-derived subgroups. Multiple views from the same subgroup remain in the same fold; the original five folds can have substantially unequal image counts.

## With local, authorized image access

The original `00_clean_make_manifest.py` was made portable: no private dataset locations are hard-coded. Pass directories that actually contain authorized TIFFs. Do not publish the generated manifest.

```bash
python angle_classical/00_clean_make_manifest.py \
  --input-dirs /PATH/TO/PRIVATE/ANGLE_IMAGES \
  --out /PATH/OUTSIDE/GIT/angle_qc

python angle_classical/01_build_feature_cache.py \
  --manifest /PATH/OUTSIDE/GIT/angle_qc/manifest_clean.csv \
  --search-spaces angle_classical/search_spaces.json \
  --out /PATH/OUTSIDE/GIT/angle_features

python angle_classical/02_stage1_select_searchspace_5fold.py \
  --features-dir /PATH/OUTSIDE/GIT/angle_features \
  --models ExtraTrees,RandomForest,LinearSVC,SGD_LogLoss,DecisionTree,LightGBM \
  --n-splits 5 --seed 42 \
  --out /PATH/OUTSIDE/GIT/angle_cv
```

Stages: `00` creates a QC manifest with group keys; `01` builds features from image pixels; `02` compares all selected classifier-feature combinations with group-aware CV, train-group weighting and held-out subgroup score aggregation.

The search space has ten explicit feature sets in `search_spaces.json` (raw gray/RGB, intensity histograms, statistics, LBP, HOG, combinations); the six manuscript classifiers are listed above. Outer independent test splitting was later used for the **CNN**, not this historical angle classical model-search stage.

**Targeted repeat:** ExtraTrees / `S09_combined_small_128` / Fold 4: image accuracy 0.4966887417, subgroup accuracy 0.6923076923, subgroup macro-F1 0.6363636364; all exactly matched saved Fold 4. Group overlap 0.

**Caveat:** Selection and reporting based on the same grouped CV can exhibit selection optimism. Describe these statistics as grouped five-fold CV, not independent external test performance. Keep manifests/results containing sample identifiers private.
