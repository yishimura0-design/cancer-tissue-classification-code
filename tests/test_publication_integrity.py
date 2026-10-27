"""Dependency-free tests for the source-only publication package.
Run from repository root: python -m unittest discover -s tests -v
These checks do not replace real-data fold-level reproducibility tests.
"""
from __future__ import annotations
import ast
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "experiment1_tissue",
    "experiment1_tissue/core_cnn_b0_a1",
    "experiment1_tissue/sensitivity_fullframe",
    "experiment2_angle_sweep_cells/angle_cnn",
    "experiment2_angle_sweep_cells/angle_classical",
    "experiment3_bf_ed_cells/bf_ed_classical",
)
BAD_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".joblib", ".pkl"}


class PublicationIntegrityTests(unittest.TestCase):
    def test_all_sources_compile(self):
        total = 0
        for dirname in SOURCE_DIRS:
            for p in (ROOT / dirname).glob("*.py"):
                ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
                total += 1
        self.assertGreaterEqual(total, 20)

    def test_excludes_private_images_and_checkpoints(self):
        for dirname in SOURCE_DIRS:
            for p in (ROOT / dirname).rglob("*"):
                if p.is_file():
                    self.assertNotIn(p.suffix.lower(), BAD_SUFFIXES, p)

    def test_no_private_cluster_root_in_public_sources(self):
        for dirname in SOURCE_DIRS:
            for p in (ROOT / dirname).glob("*.py"):
                text = p.read_text(encoding="utf-8-sig")
                self.assertNotIn("/mmfs1/projects/", text, p)
                self.assertNotIn("-----BEGIN OPENSSH PRIVATE KEY-----", text, p)

    def test_bfed_grouped_cv_fails_closed_without_stratified_group_kfold(self):
        # Dependency-free regression guard: grouping must never silently downgrade.
        source = (ROOT / "experiment3_bf_ed_cells" / "bf_ed_classical" / "bfed_rawpix_3fold_cv_testsize.py").read_text(encoding="utf-8")
        self.assertIn("Refusing ungrouped StratifiedKFold fallback", source)
        self.assertNotIn("Falling back to StratifiedKFold.", source)

    def test_config_schemas_are_documented(self):
        # Columns read by the training scripts (angle classical step 01 also needs image_id).
        with (ROOT / "examples" / "angle_manifest_template.csv").open(newline="") as f:
            h = next(csv.reader(f))
        self.assertTrue({"image_id", "path", "label", "group_key"}.issubset(h))
        # The core-level classical baseline also needs filename (validate_manifest).
        with (ROOT / "examples" / "tissue_manifest_template.csv").open(newline="") as f:
            h = next(csv.reader(f))
        self.assertTrue({"filepath", "relative_path", "filename", "sample_id_from_folder",
                         "subgroup_id", "scenario1_label", "scenario2_label"}.issubset(h))


if __name__ == "__main__":
    unittest.main()
