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
    "bf_ed_classical",
    "tissue_classical",
    "angle_classical",
    "core_cnn_b0_a1",
    "angle_cnn",
    "fullframe_padding",
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

    def test_config_schemas_are_documented(self):
        with (ROOT / "examples" / "angle_manifest_template.csv").open(newline="") as f:
            h = next(csv.reader(f))
        self.assertEqual(h, ["path", "label", "group_key"])
        with (ROOT / "examples" / "tissue_manifest_template.csv").open(newline="") as f:
            h = next(csv.reader(f))
        self.assertTrue({"subgroup_id", "sample_id_from_folder", "scenario1_label", "scenario2_label"}.issubset(h))


if __name__ == "__main__":
    unittest.main()
