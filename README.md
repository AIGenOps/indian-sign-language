# ISL Translator — Production Engineering Repository

Production-oriented Indian Sign Language (ISL) video-to-English translation system.

## Objective
Camera/video -> visual and pose features -> continuous sign recognition -> English translation -> optional speech.

## Initial pretrained model
Sanket - ISL Translator:
https://huggingface.co/ayush2635/sanket-isl-translator

The published model card describes a Dual-Stream Gated Multi-modal Transformer with TimeSformer, MediaPipe pose features, and T5-base. It reports a peak validation BLEU-1 of 50.6%. This is a published baseline, not a production acceptance result.

## Important
The Sanket checkpoint is a custom `.pth` artifact. Do not fabricate a generic Hugging Face loader. The repository includes an adapter boundary and a downloader; Prompt 02 in `docs/20_PROMPTS.md` pins the exact upstream preprocessing/model implementation before inference is enabled.

**Update, 2026-09-27:** the upstream Hugging Face repo is confirmed MIT-licensed,
but a search turned up no public source repository for the training/preprocessing
code — the model card's own GitHub link is an illustrative placeholder, not a
real URL. See `MODEL_DOWNLOAD_STATUS.md` and `docs/20_PROMPTS.md` for the
fallback plan (`scripts/inspect_state_dict.py`) and for a MediaPipe API
compatibility risk (`mediapipe.solutions` was removed in newer MediaPipe
releases) that directly affects whether the pose-feature preprocessing can be
reproduced at all.

> [!WARNING]
> **EXPERIMENTAL PIPELINE:** An experimental inference pipeline is available via explicit opt-in (`allow_experimental_inference=True`). Preprocessing parameters (Candidate A 195-dim MediaPipe pose landmarks, uniform 8-frame sampling, ImageNet normalization) are UNVERIFIED assumptions. See `docs/19_DECISION_LOG.md` **ADR-008** for full technical details and disclaimers.


## Data
Primary research resources:
- ISLTranslate: continuous ISL-English sentence/phrase data.
- INCLUDE: isolated ISL recognition.
- CISLR: large-vocabulary ISL recognition.
- ST-GCN + Transformer + CTC implementation for an alternative continuous recognition baseline.

## Principles
- production-grade architecture from the beginning
- signer-disjoint evaluation
- explicit uncertainty
- no fabricated output
- versioned model/data provenance
- privacy-first video handling
- reproducible experiments
- replaceable model provider
