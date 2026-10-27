# Full-frame CNN padding sensitivity experiment

This is an additional image preprocessing comparison, not a new independent cohort. It compares the historical crop/stretch processing to preserving the **complete field of view** in 224×224 images via aspect-ratio-preserving bicubic resize and neutral-gray letterbox padding (RGB 127/127/127). Model training retains normalizations and training-time augmentations appropriate for Core / Angle.

## Code and provenance

* `prepare_fullframe_repeat.py`: generates original-TIFF-based full-frame source copies, 57 job definitions and provenance.
* `prepare_shared_cache.py`: safely preprocesses authorized source TIFFs **once** into PNG cache outside Git, preserving source-path columns internally for labels/group membership.
* `build_cached_campaign.py`: generates 57-job cached version of Core B0/A1 + Angle Sweep CNN with same grouping and result contracts.
* `preflight_fullframe_repeat.py`: checks data/roles and network-independent readiness on the actual compute environment.
* `summarize_fullframe_repeat.py`: joins/validates result artifacts.
* `pbs_* / submit_cached_campaign.sh`: **CCast-specific templates**, not a portable local GPU submission guarantee.

## Data-free code-generator check

From repository root, use output directories outside the repo:

```bash
python experiment1_tissue/sensitivity_fullframe/prepare_fullframe_repeat.py \
  --repo . --output /tmp/fullframe_source_validation

python experiment1_tissue/sensitivity_fullframe/build_cached_campaign.py \
  --repo . --output /tmp/fullframe_cache_validation
```

Expected messages: `Generated 57 training jobs` and `CACHED_CAMPAIGN_BUILD_PASS: 57 jobs`. On the publication preparation machine on 2026-10-20, both checks passed, **without accessing any microscopy data or using GPUs**.

With authorized original images, run preprocessing and training using locally adapted paths and the validated fold manifests. Do not execute HPC submission scripts without updating your institutional queue, filesystem and CUDA settings.

**Caveat:** Source PNGs are lossless containers **after resizing**, not an invertible copy of the original TIFF image resolution. Test/Train inputs preserve the full image area but not original resolution. Original reported results are from the private CCast campaign, not from an independently accessible public dataset.
