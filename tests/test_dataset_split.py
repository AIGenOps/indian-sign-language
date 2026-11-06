"""Unit tests for dataset split generation and manifest determinism."""

import csv
import json
import pytest
from pathlib import Path
from src.dataset.split import generate_signer_disjoint_split, save_split_manifest


@pytest.fixture
def dummy_dataset_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "test_isl.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "text"])
        # Group 1: 10 clips
        for i in range(1, 11):
            writer.writerow([f"g1-{i}", f"Text sentence {i}"])
        # Group 2: 5 clips
        for i in range(1, 6):
            writer.writerow([f"g2-{i}", f"Text sentence {i}"])
        # Group 3: 5 clips
        for i in range(1, 6):
            writer.writerow([f"g3-{i}", f"Text sentence {i}"])
        # Group 4: 2 clips
        for i in range(1, 3):
            writer.writerow([f"g4-{i}", f"Text sentence {i}"])

    return csv_file


def test_split_determinism_and_schema(dummy_dataset_csv: Path, tmp_path: Path):
    manifest1 = generate_signer_disjoint_split(
        metadata_path=dummy_dataset_csv,
        val_ratio=0.25,
        test_ratio=0.25,
        max_group_share_cap=0.50,
    )
    manifest2 = generate_signer_disjoint_split(
        metadata_path=dummy_dataset_csv,
        val_ratio=0.25,
        test_ratio=0.25,
        max_group_share_cap=0.50,
    )

    # Assert exact determinism across multiple runs
    assert manifest1 == manifest2

    # Check required metadata fields
    assert "diversity_note" in manifest1["metadata"]
    assert "presenter diversity is bounded" in manifest1["metadata"]["diversity_note"].lower()
    assert manifest1["metadata"]["disjointness_assertion_result"] == "PASSED_100_PERCENT_DISJOINT"


    # Save to disk and check file reading
    out_path = tmp_path / "manifest.json"
    save_split_manifest(manifest1, out_path)
    assert out_path.exists()

    with open(out_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == manifest1
