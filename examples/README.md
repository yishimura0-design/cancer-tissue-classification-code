# Example image tables (NO REAL DATA)

These files show the format of the image table ("manifest") that step 1 of Experiments 1 and 2 produces automatically. You do not need to write these tables by hand.

The example rows were produced by the step-1 scripts themselves, run on three placeholder images per experiment. Only the image location was replaced with `/PATH/TO/...`; no real image paths or data are included.

## `tissue_manifest_template.csv` — Experiment 1

Produced by `experiment1_tissue/01_qc_make_manifest.py` (one row per accepted image). The columns the training scripts need are:

| Column | Meaning |
|---|---|
| `filepath` | Full path to the image |
| `relative_path` | Path below the tissue image folder |
| `filename` | Image file name |
| `sample_id_from_folder` | Tissue core (the unit that is kept together in one fold) |
| `subgroup_id` | Field of view: all angle images of one field of view share it |
| `scenario1_label` | Normal / Malignant |
| `scenario2_label` | Normal / Stage IB / IIA / IIB / III / IV |

The remaining columns (`suffix`, `is_tif`, `top_group_dir`, `filename_sample_tokens`, `structure_note`, `angle_value`, `qc_status`, `qc_reason`, `quarantined_to`) record the quality-control checks. In the example, the first two rows are two angles of the same field of view (same `subgroup_id`); the third row is a different core.

## `angle_manifest_template.csv` — Experiment 2

Produced by `experiment2_angle_sweep_cells/angle_classical/00_clean_make_manifest.py`. The columns the training scripts need are:

| Column | Meaning |
|---|---|
| `image_id` | Identifier computed from the image path |
| `path` | Full path to the image |
| `label` | HPNE / PANC / MIA |
| `group_key` | Subgroup (angle series): all angle images of one series share it |

`filename`, `stem`, `extension` and `parent_dir` are descriptive. In the example, the first two rows are two angles of the same series (same `group_key`); the third row is a different series.

## Experiment 3 (BF / ED)

No table is needed. The script pairs BF and ED images directly from the folder layout described in `experiment3_bf_ed_cells/bf_ed_classical/README.md`.
