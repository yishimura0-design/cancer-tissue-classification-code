# Example manifest schemas (NO REAL DATA)

These templates intentionally contain only header columns; no image paths or biological sample IDs have been copied from private experiments.

**Angle:** `angle_manifest_template.csv` uses `path,label,group_key`; `HPNE`, `PANC`, `MIA` must be valid labels and all angular views of one biological subgroup share exactly one group key.

**Tissue:** `tissue_manifest_template.csv` uses `filepath,relative_path,sample_id_from_folder,subgroup_id,scenario1_label,scenario2_label`. `filepath` must point to an authorized local image. `sample_id_from_folder` identifies the 45-core unit for Core CNN; `subgroup_id` identifies the 225-subgroup unit in historical classical ML.

**BF/ED:** no manifest is needed for the original pairing script; instead use the documented filename directory layout and set `CANCER_DATA_ROOT`.

Templates are not datasets. Obtaining actual images and validating sample-level ownership/identifiers are responsibilities of the data owner.
