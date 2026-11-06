# Coding-Agent Prompts

## Prompt 01 — Environment and artifact
Inspect the repository and implement robust diagnostics for Python, PyTorch, CUDA/GPU, MediaPipe, Transformers and OpenCV. Implement model download and checksum recording. Do not fabricate model output. Keep model logic behind ModelProvider.

## Prompt 02 — Reproduce Sanket
Read the Sanket model card and referenced implementation. Pin the exact preprocessing pipeline and source revision. Load model_best.pth. Run a deterministic sample. Record output, latency, software versions, GPU and artifact hash. If the upstream implementation cannot be pinned, stop and report the exact blocker rather than inventing a loader.

Status as of 2026-09-27: a live check of https://huggingface.co/ayush2635/sanket-isl-translator
and a web search for any companion source repository found none — the model
card's own GitHub link is an illustrative "e.g." placeholder, not a real URL.
Do not spend further cycles trying to clone it. Instead:
1. Run `scripts/inspect_state_dict.py` against the downloaded checkpoint to
   recover real parameter names/shapes for the vision encoder, pose encoder,
   fusion block and decoder.
2. Cross-check the pose encoder's input width against the "195 MediaPipe
   features/frame" claim.
3. If the exact frame-sampling rate, normalization, and MediaPipe
   landmark-to-feature mapping still cannot be determined, report that as the
   explicit blocker (per this prompt's own instruction) rather than guessing.

## Prompt 02b — MediaPipe API migration check
Before upgrading the `mediapipe` pin past 0.10.21 (see requirements.txt),
verify whether the newer Tasks API (`mediapipe.tasks.vision`) can be
configured to reproduce the same landmark set and feature ordering as the
legacy `mediapipe.solutions` API this project currently targets. Do this as
its own verification step with a fixed test video and numeric diff of the
feature vectors — do not assume parity.

## Prompt 03 — Streaming inference
Implement camera -> bounded frame buffer -> preprocessing -> model -> prediction -> stabilization. Add reconnect handling, latency metrics, confidence thresholds and explicit errors.

## Prompt 04 — Continuous benchmark
Integrate ISLTranslate as an external dataset. Build signer-disjoint splits and WER/BLEU/ROUGE evaluation. Never commit downloaded videos.

## Prompt 05 — Alternative model
Integrate an ST-GCN + Transformer + CTC model behind the same ModelProvider interface. Compare quality, latency, memory and robustness.

## Prompt 06 — API
Implement FastAPI session/frame endpoints with schemas, request limits, authentication hooks, health/readiness and structured logging.

## Prompt 07 — UI
Build an accessibility-focused interface with camera, current/confirmed translation, uncertainty state, speech toggle, privacy status and model status.

## Prompt 08 — MLOps
Implement model manifest, SHA-256, evaluation artifacts, model states, promotion metadata and rollback.

## Prompt 09 — Security
Run dependency audit, secret scan, malformed media tests, auth tests, path traversal tests and model integrity tests.

## Prompt 10 — Release
Generate SBOM, test report, model card, dataset registry, deployment manifest and rollback procedure. Never mark production-ready unless acceptance gates pass.
