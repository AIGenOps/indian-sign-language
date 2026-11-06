"""Introspect the Sanket checkpoint's raw state_dict.

Why this script exists (added 2026-09-27):
docs/20_PROMPTS.md Prompt 02 calls for pinning "the exact upstream
preprocessing/model implementation" before enabling inference. A search of
Hugging Face and GitHub (see docs/21_RESEARCH_REFERENCES.md) found no public
source repository for this checkpoint -- the model card's own "Clone the Main
GitHub Repository (e.g., https://github.com/sanket/Sanket-ISL-Translator.git)"
line is an illustrative placeholder, not a real, resolvable URL, and no other
public mirror of the training code was found.

Because of that, pinning an upstream implementation may not be possible at
all. This script is the fallback: it loads only the state_dict (never
`weights_only=False`, so no arbitrary code from the checkpoint executes) and
prints every parameter name, shape and dtype. That is enough to reconstruct
the module boundaries (TimeSformer encoder, a 2-layer pose TransformerEncoder,
a gated fusion block, a T5-base decoder) named in the model card, and to
confirm or refute the "195 MediaPipe features/frame" pose input width from
the pose encoder's first-layer input shape -- without fabricating a loader.

This does NOT replace finding real upstream code if it surfaces later; it is
a legitimate, inspection-only alternative when it does not.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch


def load_state_dict(path: Path) -> "OrderedDict[str, torch.Tensor]":
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise SystemExit(
        "Unrecognized checkpoint structure: top-level object is not a dict. "
        "Re-run with weights_only=True disabled only in a sandboxed environment "
        "you trust, and only if you understand the pickle-execution risk."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="List parameter names/shapes in a Sanket checkpoint")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/models/model_best.pth")
    parser.add_argument(
        "--filter",
        default=None,
        help="Only print parameter names containing this substring (e.g. 'pose', 'decoder').",
    )
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        raise SystemExit(f"Missing checkpoint: {path}. Run scripts/download_sanket.py first.")

    state_dict = load_state_dict(path)

    raw_total_params = 0
    dedup_total_params = 0
    seen_storages: set[tuple[int, int]] = set()

    top_level_prefixes_raw: dict[str, int] = {}
    top_level_prefixes_dedup: dict[str, int] = {}

    for name, tensor in state_dict.items():
        if args.filter and args.filter not in name:
            continue
        n = tensor.numel()
        raw_total_params += n
        prefix = name.split(".")[0]
        top_level_prefixes_raw[prefix] = top_level_prefixes_raw.get(prefix, 0) + n

        storage_key = (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        if storage_key not in seen_storages:
            seen_storages.add(storage_key)
            dedup_total_params += n
            top_level_prefixes_dedup[prefix] = top_level_prefixes_dedup.get(prefix, 0) + n

        print(f"{name:<80} {tuple(tensor.shape)!s:<24} {tensor.dtype}")

    print()
    print("Top-level module prefixes (raw key-sum):")
    for prefix, count in sorted(top_level_prefixes_raw.items(), key=lambda kv: -kv[1]):
        print(f"  {prefix:<30} {count:,}")
    print()
    print("Top-level module prefixes (deduplicated):")
    for prefix, count in sorted(top_level_prefixes_dedup.items(), key=lambda kv: -kv[1]):
        print(f"  {prefix:<30} {count:,}")
    print()
    print(f"Total tensors inspected: {len(state_dict)}")
    print(f"Total parameters (raw key-sum): {raw_total_params:,}")
    print(f"Total parameters (deduplicated): {dedup_total_params:,}")
    print()
    print("Cross-check against the published architecture in docs/16_MODEL_CARD.md:")
    print("  - a facebook/timesformer-base-finetuned-k400 vision encoder")
    print("  - a custom 2-layer TransformerEncoder pose stream (expects 195 features/frame)")
    print("  - a gated cross-modal fusion block")
    print("  - a t5-base (~220M param) language decoder")


if __name__ == "__main__":
    main()
