"""Download the Sanket ISL Translator checkpoint(s) from Hugging Face.

Updated 2026-09-27: this now uses `huggingface_hub.hf_hub_download` instead
of a raw `urllib` GET against a `resolve/main/...?download=true` URL. The
upstream repo (ayush2635/sanket-isl-translator) stores its files behind the
Hub's xet-backed LFS storage; a bare urllib request against the resolve URL
can redirect through auth/LFS pointer responses that urllib does not follow
the way a browser or the official client does, and it gives no resume,
retry, or integrity checking. `huggingface_hub` handles all of that and is
already a project dependency (see requirements.txt).

Confirmed 2026-09-27 directly against the Hub:
  - Repo: https://huggingface.co/ayush2635/sanket-isl-translator (license: mit)
  - model_best.pth        ~1.44 GB  (best checkpoint by validation loss)
  - checkpoint_latest.pth ~3.67 GB  (weights + optimizer + scheduler state)
No official GitHub source repository for the preprocessing/model code could
be found (the model card's own link is an illustrative "e.g." placeholder,
not a real URL) -- see MODEL_DOWNLOAD_STATUS.md and docs/20_PROMPTS.md
Prompt 02 before assuming inference can be enabled from this download alone.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ayush2635/sanket-isl-translator"
OUT_DIR = Path("artifacts/models")

FILES = {
    "best": "model_best.pth",
    "latest": "checkpoint_latest.pth",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(filename: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / filename
    if dest.exists():
        print(f"Already exists: {dest}")
        print("SHA256:", sha256(dest))
        return dest

    print(f"Downloading {filename} from {REPO_ID} ...")
    cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    shutil.copyfile(cached_path, dest)
    print("Downloaded:", dest)
    digest = sha256(dest)
    print("SHA256:", digest)
    print("Record this hash in configs/model_registry.example.json before deployment.")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Sanket ISL Translator checkpoint(s)")
    parser.add_argument(
        "--which",
        choices=["best", "latest", "both"],
        default="best",
        help="'best' (model_best.pth, ~1.44GB, default) is enough to run inference. "
        "'latest' (checkpoint_latest.pth, ~3.67GB) also includes optimizer/scheduler "
        "state and is only needed to resume/fine-tune training.",
    )
    args = parser.parse_args()

    targets = ["best", "latest"] if args.which == "both" else [args.which]
    for key in targets:
        fetch(FILES[key])

    print()
    print("License: the upstream Hugging Face repo is tagged 'mit'. Re-verify at")
    print(f"https://huggingface.co/{REPO_ID} before redistributing the weights, since")
    print("model cards can be edited after the fact.")


if __name__ == "__main__":
    main()
