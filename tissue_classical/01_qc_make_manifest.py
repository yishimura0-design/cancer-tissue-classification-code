#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a clean, paper-ready manifest for the Cancer_tissue dataset.

Expected original image layout:
    Cancer_tissue/A/A1/A1_1_20x_ang0.tif

Key QC rules:
- Only .tif/.tiff files are candidates.
- The folder-derived sample ID must agree with the filename-derived sample ID.
- Accepted filenames must match the expected angular-image naming pattern.
- Known derived/intermediate processing directories are excluded.
- Wrong-label files can optionally be quarantined, but nothing is deleted.

The clean manifest includes a leakage-safe subgroup ID made by removing the
angle token while retaining the parent folder context.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCENARIO1_CLASSES = {
    "Normal": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
    "Malignant": [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
        "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
    ],
}

SCENARIO2_CLASSES = {
    "Stage IB": ["A2", "B5", "B6", "B8", "D1", "E1", "E8"],
    "Stage IIA": [
        "A4", "A5", "A6", "B1", "B2", "B7",
        "C2", "C4", "C5", "C6", "C8",
        "D2", "D3", "D5", "D7", "D8",
        "E2", "E3", "E4", "E5", "E6", "E7",
    ],
    "Stage IIB": ["A3", "A7", "B3", "B4", "C1", "D6"],
    "Stage III": ["A1", "C3"],
    "Stage IV": ["A8", "C7", "D4"],
    "Normal": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"],
}

EXPECTED_SAMPLE_IDS = [f"{group}{idx}" for group in "ABCDEF" for idx in range(1, 9)]
SAMPLE_DIR_RE = re.compile(r"^[A-F][1-8]$", re.IGNORECASE)
SAMPLE_TOKEN_RE = re.compile(r"(?<![A-Z0-9])([A-F][1-8])(?![0-9])", re.IGNORECASE)
TIF_SUFFIXES = {".tif", ".tiff"}

# Original angular image naming rule. Examples:
# A1_1_20x_ang0.tif, A1_1_20x_angle-10.tiff, A1_1_20x_ang+2.5.tif
VALID_ANGLE_TIF_RE = re.compile(
    r"^[A-F][1-8]_\d+_20x_(?:ang|angle)[_-]?[+-]?\d+(?:\.\d+)?\.(?:tif|tiff)$",
    re.IGNORECASE,
)
ANGLE_VALUE_RE = re.compile(
    r"(?:ang|angle)[_-]?([+-]?\d+(?:\.\d+)?)\.(?:tif|tiff)$",
    re.IGNORECASE,
)
ANGLE_TOKEN_AT_END_RE = re.compile(
    r"(?i)(?:[_\-\s]*)(?:ang|angle)[_\-\s]*[+-]?\d+(?:\.\d+)?$"
)

PROCESSING_DIRECTORY_NAMES = {
    "acf_180deg_float32",
    "cell_identification_v3_enhancedbf_acmasked",
}

EXPECTED_CLEAN_PROFILE = {
    "accepted_images": 16200,
    "subgroups": 225,
    "accepted_sample_ids": 45,
    "images_per_subgroup": 72,
    "scenario1": {"Malignant": 14040, "Normal": 2160},
    "scenario2": {
        "Stage IB": 2520,
        "Stage IIA": 7920,
        "Stage IIB": 2160,
        "Stage III": 720,
        "Stage IV": 720,
        "Normal": 2160,
    },
}


def invert_mapping(mapping: Dict[str, List[str]]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for label, sample_ids in mapping.items():
        for sample_id in sample_ids:
            result[sample_id.upper()] = label
    return result


SCENARIO1_BY_SAMPLE = invert_mapping(SCENARIO1_CLASSES)
SCENARIO2_BY_SAMPLE = invert_mapping(SCENARIO2_CLASSES)


def find_sample_id_from_path(path: Path, root: Path) -> Tuple[Optional[str], Optional[str], str]:
    """Return folder-derived sample ID, top group, and structure status."""
    rel_parts = path.relative_to(root).parts
    if len(rel_parts) < 3:
        return None, None, "too_shallow_expected_group_sample_file"

    group_dir = rel_parts[0].upper()
    sample_id: Optional[str] = None
    for part in rel_parts[:-1]:
        candidate = part.upper()
        if SAMPLE_DIR_RE.fullmatch(candidate):
            sample_id = candidate
            break

    if sample_id is None:
        return None, group_dir, "no_sample_dir_like_A1_to_F8"
    if group_dir not in set("ABCDEF"):
        return sample_id, group_dir, "top_group_not_A_to_F"
    if sample_id[0] != group_dir:
        return sample_id, group_dir, f"top_group_{group_dir}_does_not_match_sample_{sample_id}"
    return sample_id, group_dir, "ok"


def filename_sample_tokens(filename: str) -> List[str]:
    return sorted({m.group(1).upper() for m in SAMPLE_TOKEN_RE.finditer(filename.upper())})


def infer_subgroup_id(relative_path: str, sample_id: str, filename: str) -> str:
    """Remove only the terminal angle token and retain parent-folder context."""
    rel = Path(str(relative_path))
    parent = rel.parent.as_posix() if str(rel.parent) != "." else str(sample_id)
    stem = Path(str(filename)).stem
    subgroup_stem = ANGLE_TOKEN_AT_END_RE.sub("", stem).rstrip("_- ")
    if not subgroup_stem:
        subgroup_stem = stem
    return f"{parent}/{subgroup_stem}"


def parse_angle(filename: str) -> str:
    match = ANGLE_VALUE_RE.search(filename)
    return match.group(1) if match else ""


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_quarantine(src: Path, root: Path, quarantine_root: Path) -> Path:
    relative = src.relative_to(root)
    destination = quarantine_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        counter = 1
        while True:
            candidate = destination.with_name(
                f"{destination.stem}__dup{counter}{destination.suffix}"
            )
            if not candidate.exists():
                destination = candidate
                break
            counter += 1
    shutil.move(str(src), str(destination))
    return destination


def clean_profile_checks(
    accepted_rows: List[dict],
    sample_counter: Counter,
    scenario1_counter: Counter,
    scenario2_counter: Counter,
) -> List[str]:
    failures: List[str] = []
    subgroup_counter = Counter(row["subgroup_id"] for row in accepted_rows)

    if len(accepted_rows) != EXPECTED_CLEAN_PROFILE["accepted_images"]:
        failures.append(
            f"accepted_images={len(accepted_rows)} expected={EXPECTED_CLEAN_PROFILE['accepted_images']}"
        )
    if len(subgroup_counter) != EXPECTED_CLEAN_PROFILE["subgroups"]:
        failures.append(
            f"subgroups={len(subgroup_counter)} expected={EXPECTED_CLEAN_PROFILE['subgroups']}"
        )
    if len(sample_counter) != EXPECTED_CLEAN_PROFILE["accepted_sample_ids"]:
        failures.append(
            f"accepted_sample_ids={len(sample_counter)} expected={EXPECTED_CLEAN_PROFILE['accepted_sample_ids']}"
        )

    expected_group_size = EXPECTED_CLEAN_PROFILE["images_per_subgroup"]
    bad_group_sizes = {g: n for g, n in subgroup_counter.items() if n != expected_group_size}
    if bad_group_sizes:
        examples = list(sorted(bad_group_sizes.items()))[:10]
        failures.append(
            f"{len(bad_group_sizes)} subgroups do not contain {expected_group_size} images; examples={examples}"
        )

    if dict(scenario1_counter) != EXPECTED_CLEAN_PROFILE["scenario1"]:
        failures.append(
            f"scenario1_counts={dict(scenario1_counter)} expected={EXPECTED_CLEAN_PROFILE['scenario1']}"
        )
    if dict(scenario2_counter) != EXPECTED_CLEAN_PROFILE["scenario2"]:
        failures.append(
            f"scenario2_counts={dict(scenario2_counter)} expected={EXPECTED_CLEAN_PROFILE['scenario2']}"
        )

    key_counter = Counter((row["subgroup_id"], row["angle_value"]) for row in accepted_rows)
    duplicate_keys = [key for key, count in key_counter.items() if count > 1]
    if duplicate_keys:
        failures.append(
            f"duplicate subgroup-angle pairs={len(duplicate_keys)}; examples={duplicate_keys[:10]}"
        )

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Cancer_tissue root directory")
    parser.add_argument("--out", required=True, help="QC output directory")
    parser.add_argument(
        "--quarantine-wrong-label",
        action="store_true",
        help="Move only folder/filename sample-ID mismatches to quarantine. Nothing is deleted.",
    )
    parser.add_argument(
        "--quarantine-root",
        default=None,
        help="Default: <dataset-parent>/QC_excluded_wrong_sample_id",
    )
    parser.add_argument(
        "--strict-clean",
        action="store_true",
        help="Exit nonzero unless the expected 16,200-image/225-subgroup clean profile is met.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    quarantine_root = (
        Path(args.quarantine_root).resolve()
        if args.quarantine_root
        else root.parent / "QC_excluded_wrong_sample_id"
    )

    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    accepted_rows: List[dict] = []
    all_rows: List[dict] = []
    non_tif_rows: List[dict] = []
    excluded_rows: List[dict] = []

    status_counter: Counter = Counter()
    suffix_counter: Counter = Counter()
    sample_counter: Counter = Counter()
    subgroup_counter: Counter = Counter()
    scenario1_counter: Counter = Counter()
    scenario2_counter: Counter = Counter()
    moved_count = 0

    files = sorted(path for path in root.rglob("*") if path.is_file())

    for path in files:
        suffix = path.suffix.lower()
        suffix_counter[suffix if suffix else "<no_suffix>"] += 1
        sample_id, group_dir, structure_note = find_sample_id_from_path(path, root)
        scenario1_label = SCENARIO1_BY_SAMPLE.get(sample_id or "", "")
        scenario2_label = SCENARIO2_BY_SAMPLE.get(sample_id or "", "")
        tokens = filename_sample_tokens(path.name)
        is_tif = suffix in TIF_SUFFIXES
        relative_path = str(path.relative_to(root))

        base = {
            "filepath": str(path),
            "relative_path": relative_path,
            "filename": path.name,
            "suffix": suffix,
            "is_tif": int(is_tif),
            "top_group_dir": group_dir or "",
            "sample_id_from_folder": sample_id or "",
            "filename_sample_tokens": ";".join(tokens),
            "structure_note": structure_note,
            "scenario1_label": scenario1_label,
            "scenario2_label": scenario2_label,
            "subgroup_id": "",
            "angle_value": "",
            "qc_status": "",
            "qc_reason": "",
            "quarantined_to": "",
        }

        if not is_tif:
            base["qc_status"] = "non_tif_summarized"
            base["qc_reason"] = "not_tif_or_tiff"
            non_tif_rows.append(base.copy())
            all_rows.append(base.copy())
            status_counter[base["qc_status"]] += 1
            continue

        if sample_id is None or structure_note != "ok":
            base["qc_status"] = "exclude_bad_folder_structure"
            base["qc_reason"] = structure_note
            excluded_rows.append(base.copy())
            all_rows.append(base.copy())
            status_counter[base["qc_status"]] += 1
            continue

        relative_parts_lower = {
            part.lower() for part in path.relative_to(root).parts[:-1]
        }
        processing_hits = sorted(
            relative_parts_lower.intersection(PROCESSING_DIRECTORY_NAMES)
        )
        if processing_hits:
            base["qc_status"] = "exclude_processing_output"
            base["qc_reason"] = (
                "derived_or_intermediate_processing_directory:"
                + ";".join(processing_hits)
            )
            excluded_rows.append(base.copy())
            all_rows.append(base.copy())
            status_counter[base["qc_status"]] += 1
            continue

        if not VALID_ANGLE_TIF_RE.fullmatch(path.name):
            base["qc_status"] = "exclude_nonangular_tif"
            base["qc_reason"] = (
                "filename_does_not_match_expected_angular_image_pattern"
            )
            excluded_rows.append(base.copy())
            all_rows.append(base.copy())
            status_counter[base["qc_status"]] += 1
            continue

        foreign_tokens = [token for token in tokens if token != sample_id]
        if foreign_tokens:
            base["qc_status"] = "exclude_wrong_sample_id_in_filename"
            base["qc_reason"] = (
                f"folder_sample={sample_id};filename_has={';'.join(tokens)}"
            )
            if args.quarantine_wrong_label:
                destination = safe_quarantine(path, root, quarantine_root)
                base["quarantined_to"] = str(destination)
                moved_count += 1
            excluded_rows.append(base.copy())
            all_rows.append(base.copy())
            status_counter[base["qc_status"]] += 1
            continue

        if sample_id not in tokens:
            base["qc_status"] = "exclude_missing_matching_sample_token"
            base["qc_reason"] = (
                f"valid_angular_name_but_filename_does_not_contain_folder_sample={sample_id}"
            )
            excluded_rows.append(base.copy())
            all_rows.append(base.copy())
            status_counter[base["qc_status"]] += 1
            continue

        if not scenario1_label or not scenario2_label:
            base["qc_status"] = "exclude_sample_not_in_class_mapping"
            base["qc_reason"] = f"sample_id={sample_id} not found in scenario mappings"
            excluded_rows.append(base.copy())
            all_rows.append(base.copy())
            status_counter[base["qc_status"]] += 1
            continue

        base["subgroup_id"] = infer_subgroup_id(relative_path, sample_id, path.name)
        base["angle_value"] = parse_angle(path.name)
        base["qc_status"] = "accepted_tif"
        base["qc_reason"] = "valid_angular_image_and_folder_filename_match"

        accepted_rows.append(base.copy())
        all_rows.append(base.copy())
        status_counter[base["qc_status"]] += 1
        sample_counter[sample_id] += 1
        subgroup_counter[base["subgroup_id"]] += 1
        scenario1_counter[scenario1_label] += 1
        scenario2_counter[scenario2_label] += 1

    fieldnames = [
        "filepath",
        "relative_path",
        "filename",
        "suffix",
        "is_tif",
        "top_group_dir",
        "sample_id_from_folder",
        "filename_sample_tokens",
        "structure_note",
        "scenario1_label",
        "scenario2_label",
        "subgroup_id",
        "angle_value",
        "qc_status",
        "qc_reason",
        "quarantined_to",
    ]

    write_csv(out_dir / "manifest_accepted.csv", accepted_rows, fieldnames)
    write_csv(out_dir / "qc_all_files.csv", all_rows, fieldnames)
    write_csv(out_dir / "non_tif_files.csv", non_tif_rows, fieldnames)
    write_csv(out_dir / "excluded_or_risky_files.csv", excluded_rows, fieldnames)

    subgroup_size_rows = [
        {"subgroup_id": group, "n_images": count}
        for group, count in sorted(subgroup_counter.items())
    ]
    write_csv(
        out_dir / "subgroup_sizes.csv",
        subgroup_size_rows,
        ["subgroup_id", "n_images"],
    )

    key_counter = Counter((row["subgroup_id"], row["angle_value"]) for row in accepted_rows)
    duplicate_rows = [
        {"subgroup_id": group, "angle_value": angle, "count": count}
        for (group, angle), count in sorted(key_counter.items())
        if count > 1
    ]
    write_csv(
        out_dir / "duplicate_subgroup_angle.csv",
        duplicate_rows,
        ["subgroup_id", "angle_value", "count"],
    )

    missing_samples = [sample_id for sample_id in EXPECTED_SAMPLE_IDS if sample_counter[sample_id] == 0]
    strict_failures = clean_profile_checks(
        accepted_rows,
        sample_counter,
        scenario1_counter,
        scenario2_counter,
    )

    summary = {
        "root": str(root),
        "output_dir": str(out_dir),
        "files_total": len(files),
        "accepted_tif_total": len(accepted_rows),
        "accepted_subgroups_total": len(subgroup_counter),
        "accepted_sample_ids_total": len(sample_counter),
        "non_tif_total": len(non_tif_rows),
        "excluded_or_risky_total": len(excluded_rows),
        "moved_to_quarantine_total": moved_count,
        "status_counts": dict(status_counter),
        "suffix_counts": dict(suffix_counter),
        "accepted_counts_by_sample_id": dict(sorted(sample_counter.items())),
        "subgroup_size_counts": dict(Counter(subgroup_counter.values())),
        "missing_sample_ids_after_qc": missing_samples,
        "scenario1_class_counts": dict(scenario1_counter),
        "scenario2_class_counts": dict(scenario2_counter),
        "duplicate_subgroup_angle_total": len(duplicate_rows),
        "strict_clean_requested": bool(args.strict_clean),
        "strict_clean_pass": not strict_failures,
        "strict_clean_failures": strict_failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qc_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "Cancer_tissue QC Summary",
        "========================",
        f"Root: {root}",
        f"Total files: {len(files)}",
        f"Accepted TIF/TIFF files: {len(accepted_rows)}",
        f"Accepted subgroups: {len(subgroup_counter)}",
        f"Accepted sample IDs: {len(sample_counter)}",
        f"Non-TIF summarized: {len(non_tif_rows)}",
        f"Excluded/risky TIF files: {len(excluded_rows)}",
        f"Moved to quarantine: {moved_count}",
        "",
        "QC status counts:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in status_counter.most_common())
    lines.extend(["", "Subgroup size distribution:"])
    lines.extend(
        f"  {size} images/subgroup: {count} subgroups"
        for size, count in sorted(Counter(subgroup_counter.values()).items())
    )
    lines.extend(["", "Missing sample IDs after QC:"])
    lines.append("  " + (", ".join(missing_samples) if missing_samples else "None"))
    lines.extend(["", "Scenario 1 class counts:"])
    lines.extend(f"  {key}: {value}" for key, value in scenario1_counter.items())
    lines.extend(["", "Scenario 2 class counts:"])
    lines.extend(f"  {key}: {value}" for key, value in scenario2_counter.items())
    lines.extend(["", f"Duplicate subgroup-angle pairs: {len(duplicate_rows)}"])
    lines.extend(["", "Strict clean profile:"])
    if strict_failures:
        lines.append("  FAIL")
        lines.extend(f"  - {failure}" for failure in strict_failures)
    else:
        lines.append("  PASS")

    (out_dir / "qc_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("DONE QC")
    print(f"Accepted manifest: {out_dir / 'manifest_accepted.csv'}")
    print(f"Accepted images:   {len(accepted_rows)}")
    print(f"Accepted subgroups:{len(subgroup_counter)}")
    print(f"Summary:           {out_dir / 'qc_summary.txt'}")
    if args.quarantine_wrong_label:
        print(f"Quarantine root:   {quarantine_root}")

    if args.strict_clean and strict_failures:
        print("STRICT CLEAN CHECK FAILED:", file=sys.stderr)
        for failure in strict_failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
