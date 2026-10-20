"""Tests for the read-only B0/A1 Core and Single Image metrics exporter."""
from pathlib import Path

import pandas as pd
import pytest

import export_core_image_metrics as x


def test_pooled_balanced_accuracy_counts_minor_class():
    y = pd.Series(["Normal"] * 6 + ["Malignant"] * 39)
    predicted = pd.Series(["Normal"] * 4 + ["Malignant"] * 41)
    scores, per_class = x.describe(y, predicted, ["Normal", "Malignant"])
    assert scores["n_predictions"] == 45
    assert scores["accuracy"] == round(43 / 45, 6)
    assert scores["balanced_accuracy"] == round((4 / 6 + 39 / 39) / 2, 6)
    assert [r["correct"] for r in per_class] == [4, 39]


def test_balanced_accuracy_requires_all_classes():
    with pytest.raises(ValueError, match="missing classes"):
        x.describe(pd.Series(["Normal"]), pd.Series(["Normal"]), ["Normal", "Malignant"])


def fake_fold(tmp_path: Path, model="resnet18"):
    root = tmp_path / "results"
    d = root / "B0" / "cnn" / "scenario1" / model / "fold_1"
    d.mkdir(parents=True)
    test_cores = [f"Z{i}" for i in range(9)]
    roles = pd.DataFrame({
        "outer_fold": [1] * 9,
        "core_id": test_cores,
        "role": ["test"] * 9,
        "scenario1_label": ["Normal"] * 2 + ["Malignant"] * 7,
        "scenario2_label": ["Normal"] * 2 + ["Stage IIA"] * 7,
    })
    core = roles[["core_id"]].copy()
    core["target_label"] = roles["scenario1_label"]
    core["prediction_label"] = roles["scenario1_label"]
    rows = []
    for core_id, label in zip(roles.core_id, roles.scenario1_label):
        for fov in range(5):
            for angle in range(72):
                rows.append({
                    "path": f"/not_real/{core_id}/{fov}/{angle}.tif",
                    "core_id": core_id,
                    "fov_id": f"{core_id}_{fov}",
                    "target_label": label,
                    "prediction_label": label,
                    "experiment_id": "B0",
                    "scenario": "scenario1",
                    "model": model,
                    "fold": 1,
                })
    image = pd.DataFrame(rows)
    image.to_csv(d / "test_image_predictions.csv.gz", index=False, compression="gzip")
    return root, roles, core, image, d


def test_validated_image_fold_and_a1_reuse(tmp_path, monkeypatch):
    root, roles, core, _, _ = fake_fold(tmp_path)
    monkeypatch.setattr(x, "cnn_fold", lambda *args: core)
    result = x.validated_fold(root, roles, "B0", "scenario1", "resnet18", 1)
    assert len(result[0]) == 9
    assert len(result[1]) == 3240
    # Unchanged A1 Fold 1 must reuse B0 prediction artifacts, not require an A1 copy.
    reused = x.validated_fold(root, roles, "A1", "scenario1", "resnet18", 1)
    assert len(reused[1]) == 3240


def test_rejects_image_truth_mismatch(tmp_path, monkeypatch):
    root, roles, core, image, d = fake_fold(tmp_path)
    monkeypatch.setattr(x, "cnn_fold", lambda *args: core)
    image.loc[0, "target_label"] = "Malignant"
    image.to_csv(d / "test_image_predictions.csv.gz", index=False, compression="gzip")
    x.read_image_csv.cache_clear()
    with pytest.raises(ValueError, match="ground truth differs"):
        x.validated_fold(root, roles, "B0", "scenario1", "resnet18", 1)


def test_rejects_duplicate_image_path(tmp_path, monkeypatch):
    root, roles, core, image, d = fake_fold(tmp_path)
    monkeypatch.setattr(x, "cnn_fold", lambda *args: core)
    image.loc[0, "path"] = image.loc[1, "path"]
    image.to_csv(d / "test_image_predictions.csv.gz", index=False, compression="gzip")
    x.read_image_csv.cache_clear()
    with pytest.raises(ValueError, match="3,240 unique paths"):
        x.validated_fold(root, roles, "B0", "scenario1", "resnet18", 1)


def test_incomplete_experiments_have_no_invented_scores(tmp_path):
    repo = Path(__file__).resolve().parent
    result = x.summarize(
        tmp_path / "not_started",
        repo / "splits" / "inner_outer_roles.csv",
        repo / "splits_variant_a1" / "inner_outer_roles.csv",
        tmp_path / "aggregates",
    )
    assert len(result["models"]) == 16
    assert all(r["status"] == "incomplete" for r in result["models"])
    output = pd.read_csv(tmp_path / "aggregates" / "core_image_metrics.csv")
    assert output.core_balanced_accuracy.isna().all()
    assert output.single_image_accuracy.isna().all()
