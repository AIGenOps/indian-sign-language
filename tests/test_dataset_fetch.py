"""Unit tests for dataset fetching and parsing module."""

import csv
import pytest
from pathlib import Path
from src.dataset.fetch import parse_source_video_id, load_isl_translate_records


def test_parse_source_video_id():
    assert parse_source_video_id("1782bea75c7d-3") == "1782bea75c7d"
    assert parse_source_video_id("abc1234-99") == "abc1234"
    assert parse_source_video_id("nohyphen") == "nohyphen"
    assert parse_source_video_id("multi-part-id-1") == "multi"


def test_load_isl_translate_records(tmp_path: Path):
    csv_file = tmp_path / "test_isl.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "text"])
        writer.writerow(["vid001-1", "Hello world"])
        writer.writerow(["vid001-2", "How are you?"])
        writer.writerow(["vid002-1", "Thank you"])

    records = load_isl_translate_records(csv_file)
    assert len(records) == 3
    assert records[0] == {
        "uid": "vid001-1",
        "text": "Hello world",
        "source_video_id": "vid001",
    }
    assert records[1]["source_video_id"] == "vid001"
    assert records[2]["source_video_id"] == "vid002"
