# Initial Model Card

## Model
Sanket - ISL Translator

Source:
https://huggingface.co/ayush2635/sanket-isl-translator

## Architecture
TimeSformer visual encoder + MediaPipe pose/skeleton encoder + gated multimodal fusion + T5-base decoder.

## Published training information
INCLUDE dataset, approximately 4,284 videos across 250 categories in the model card, two-phase training, AWS A10G environment.

## Published result
Peak validation BLEU-1: 50.6%.

This repository does not independently reproduce that number.

## Verified facts (checked live against the Hugging Face repo, 2026-09-27)
- License tag: **mit**.
- Files: `model_best.pth` (~1.44 GB, best checkpoint by validation loss) and
  `checkpoint_latest.pth` (~3.67 GB, includes optimizer/scheduler state),
  both xet/LFS-backed.
- **No public source repository for the training/preprocessing code was
  found.** The model card's "Clone the Main GitHub Repository (e.g.,
  `https://github.com/sanket/Sanket-ISL-Translator.git`)" instruction is an
  illustrative placeholder, not a working link. Treat the exact preprocessing
  contract (frame sampling rate, MediaPipe feature composition, tokenizer
  config) as unknown until proven otherwise via `scripts/inspect_state_dict.py`
  or a future authoritative release from the author.
- The "195 MediaPipe features/frame" pose stream was very likely produced
  with MediaPipe's legacy `mediapipe.solutions` Holistic/Pose/Hands API, which
  newer MediaPipe releases (past 0.10.21) no longer expose. See
  `MODEL_DOWNLOAD_STATUS.md` and `requirements.txt`.

## Intended use
Initial engineering baseline for ISL video-to-English translation.

## Limitations
Dataset coverage, real-world continuous conversation, signer diversity and exact preprocessing fidelity require independent validation. The lack of a public source repository (above) makes preprocessing fidelity the single biggest open risk for this baseline.

## Approval
CANDIDATE / BASELINE. Not production-approved.

## EXPERIMENTAL PIPELINE NOTICE
An experimental inference pipeline is implemented behind `allow_experimental_inference=True` in `SanketModelProvider`. See `docs/19_DECISION_LOG.md` **ADR-008** for details. Preprocessing parameters (Candidate A 195-dim MediaPipe pose landmarks, uniform 8-frame sampling, ImageNet normalization) are UNVERIFIED assumptions and outputs MUST NOT be interpreted as a reproduction of the published BLEU-1 50.6 baseline.

