# Model Binary Status

The ZIP contains the model acquisition mechanism, model adapter, documentation and exact source reference.

The Sanket `model_best.pth` binary is NOT embedded because the package build environment could not resolve the external Hugging Face host.

Download it with:

```bash
python scripts/download_sanket.py            # model_best.pth only (default, ~1.44 GB)
python scripts/download_sanket.py --which latest   # checkpoint_latest.pth (~3.67 GB, resume/fine-tune)
python scripts/download_sanket.py --which both
```

Source:
https://huggingface.co/ayush2635/sanket-isl-translator

Output:
`artifacts/models/model_best.pth` (and/or `checkpoint_latest.pth`)

Then:
```bash
python scripts/inspect_model_artifact.py
python scripts/inspect_state_dict.py            # layer-level introspection, see below
```

Record the SHA-256 in the model registry (`configs/model_registry.example.json`).

## Verified against the live Hugging Face repo (checked 2026-09-27)
- License tag on the repo: **mit**. Re-check before redistribution, since model
  cards can be edited after the fact — this file is a point-in-time record, not
  a legal guarantee.
- Files present: `model_best.pth` (~1.44 GB) and `checkpoint_latest.pth` (~3.67 GB),
  both stored via the Hub's xet/LFS backend, plus a README and `.gitattributes`.
  `scripts/download_sanket.py` now uses `huggingface_hub.hf_hub_download` rather
  than a raw URL fetch for this reason.
- **No public source repository was found.** The model card's own instructions
  say "Clone the Main GitHub Repository (e.g.,
  `https://github.com/sanket/Sanket-ISL-Translator.git`)" — that URL is an
  illustrative placeholder ("e.g."), not a real, resolvable repository, and no
  other public mirror of the training/preprocessing code turned up in search.

  State-dict introspection (`scripts/inspect_state_dict.py`) confirmed the exact
  model parameter shapes and architecture boundaries for `model_best.pth`:

  | Preprocessing / Model Contract Fact | Value / Specification | Status |
  | :--- | :--- | :--- |
  | **Frames per Clip** | `8` frames | **CONFIRMED FROM WEIGHTS** (`time_embeddings` shape `(1, 8, 768)`) |
  | **Input Resolution** | `224x224` RGB | **CONFIRMED FROM WEIGHTS** (`position_embeddings` `197` tokens -> 14x14 grid) |
  | **Patch Size** | `16x16` pixels | **CONFIRMED FROM WEIGHTS** (`patch_embeddings.projection` kernel `(16, 16)`) |
  | **Pose Vector Dimension** | `195` floats / frame | **CONFIRMED FROM WEIGHTS** (`pose_encoder.fc_in` shape `(768, 195)`) |
  | **MediaPipe Landmark Set / Feature Mapping** | Exact landmark selection & ordering producing 195 dims | **UNKNOWN / OPEN RISK** (Vector length 195 confirmed, but exact landmark subset/ordering is unlisted) |
  | **Image Normalization Values** | Mean & Std (e.g. ImageNet `[0.45, 0.45, 0.45]`, `[0.225, 0.225, 0.225]`) | **ASSUMED FROM MODEL CARD (UNVERIFIED)** |
  | **Tokenizer & Vocab Size** | `T5-base` tokenizer (`32,128` vocabulary) | **CONFIRMED FROM WEIGHTS** (`text_decoder.shared` shape `(32128, 768)`), special tokens/prompts **ASSUMED FROM MODEL CARD (UNVERIFIED)** |
  | **Frame Sampling Stride / Rate** | FPS or temporal sampling policy across raw video | **UNKNOWN / OPEN RISK** (No temporal stride metadata in checkpoint) |

  **CRITICAL OPEN RISK:** The exact MediaPipe landmark selection, ordering, and coordinate representation that produces the 195-dim pose vector remain **unrecoverable without upstream source code**. Any pose preprocessing module built at this stage (e.g., `src/preprocessing/pose.py`) is explicitly classified as **EXPERIMENTAL / UNVERIFIED**. Do not claim reproduction of the original baseline performance until this preprocessing contract is independently verified.

- **MediaPipe risk**: the model card describes a pose stream built from "195
  MediaPipe features/frame". MediaPipe removed its legacy `mediapipe.solutions`
  API (Holistic/Pose/Hands landmarkers, the thing most ISL pose-feature
  pipelines of this era were built on) in favor of the newer Tasks API
  somewhere between versions 0.10.21 and 0.10.31. `requirements.txt` pins
  `mediapipe==0.10.21` (the last version confirmed to still expose
  `mediapipe.solutions`) so any attempt to reproduce the original feature
  vector at least has the right API surface available. Do not upgrade that
  pin without first confirming which landmarker set and which 195 dimensions
  the original authors actually used.

Do not redistribute the checkpoint until its current terms have been reviewed independently of this document.
