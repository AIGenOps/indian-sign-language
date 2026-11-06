"""Module for fetching and caching ISLTranslate dataset metadata and records.

Subject to CC BY-NC-SA 4.0 license (Exploration-Lab/ISLTranslate).
Raw video data and fetched CSV metadata remain strictly gitignored locally.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_ISL_TRANSLATE_CSV_URL = (
    "https://raw.githubusercontent.com/Exploration-Lab/ISLTranslate/main/data/ISLTranslate.csv"
)
DEFAULT_METADATA_PATH = Path("data/isl_translate.csv")


def download_isl_translate_metadata(
    target_path: Path = DEFAULT_METADATA_PATH,
    url: str = DEFAULT_ISL_TRANSLATE_CSV_URL,
    force_redownload: bool = False,
) -> Path:
    """Download and cache ISLTranslate metadata CSV.

    Args:
        target_path: Local destination path for the CSV.
        url: Remote URL for ISLTranslate.csv.
        force_redownload: If True, redownload even if file exists.

    Returns:
        Path to the downloaded CSV file.

    Raises:
        RuntimeError: If download fails or CSV format is invalid.
    """
    target_path = Path(target_path)
    if target_path.exists() and not force_redownload:
        logger.info(f"Dataset metadata already cached at {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading ISLTranslate metadata from {url} to {target_path}...")

    req = urllib.request.Request(url, headers={"User-Agent": "ISL-Translator-Benchmark/1.0"})
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode("utf-8")
        
        target_path.write_text(content, encoding="utf-8")
        logger.info(f"Successfully downloaded metadata ({len(content)} bytes).")
    except Exception as e:
        raise RuntimeError(f"Failed to download ISLTranslate metadata from {url}: {e}") from e

    # Validation check on downloaded content
    records = load_isl_translate_records(target_path)
    if not records:
        raise RuntimeError(f"Downloaded CSV at {target_path} contained 0 valid records.")

    logger.info(f"Verified dataset metadata with {len(records)} entries.")
    return target_path


def parse_source_video_id(uid: str) -> str:
    """Extract source_video_id prefix from clip uid.

    Example: '1782bea75c7d-3' -> '1782bea75c7d'
    If no hyphen present, returns the full uid.
    """
    if "-" in uid:
        return uid.split("-", 1)[0]
    return uid


def load_isl_translate_records(csv_path: Path = DEFAULT_METADATA_PATH) -> List[Dict[str, str]]:
    """Load records from ISLTranslate CSV file.

    Returns:
        List of dicts with keys: 'uid', 'text', 'source_video_id'
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {csv_path}. Call download_isl_translate_metadata first.")

    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "uid" not in reader.fieldnames:
            raise ValueError(f"CSV header invalid. Expected 'uid' column, got: {reader.fieldnames}")

        text_col = "text" if "text" in reader.fieldnames else ("translation" if "translation" in reader.fieldnames else None)
        if not text_col:
            raise ValueError(f"CSV header invalid. Expected 'text' or 'translation' column, got: {reader.fieldnames}")

        for row in reader:
            uid = row["uid"].strip()
            text = row[text_col].strip()
            if not uid:
                continue
            source_video_id = parse_source_video_id(uid)
            records.append({
                "uid": uid,
                "text": text,
                "source_video_id": source_video_id,
            })

    return records
