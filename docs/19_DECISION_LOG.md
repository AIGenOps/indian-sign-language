# Architecture Decision Records

## ADR-001
Start with Sanket because it explicitly targets ISL video-to-English translation and provides trained weights.

Status: baseline only.

## ADR-002
Use a model-provider abstraction so the final production model can be changed without rewriting the product.

## ADR-003
Prefer edge-first processing because continuous signer video is sensitive.

## ADR-004
Pin every dependency in `requirements.txt` with both a floor and a ceiling
(2026-09-27 update pass), rather than floors only. Rationale: a live check of
current releases found major-version jumps in flight for several deps this
project touches directly (transformers 4.x -> 5.x, huggingface_hub 0.x -> 1.x,
mediapipe 0.10.x -> 1.0.x with `mediapipe.solutions` removed along the way,
numpy 1.x -> 2.x, pandas 2.x -> 3.x, pytest 8.x -> 9.x). Because this project's
whole premise is byte-for-byte-plausible reproduction of someone else's
preprocessing, an unpinned ceiling risks silently changing behavior on a
routine `pip install -r requirements.txt`. Ceilings should be revisited
deliberately, with a test run, not bumped reflexively.

## ADR-005
`scripts/download_sanket.py` fetches through `huggingface_hub.hf_hub_download`
rather than a raw URL GET. Rationale: the upstream repo's files are
xet/LFS-backed; the official client handles pointer resolution, resume and
retries correctly, a bare `urllib` request is not guaranteed to.

## ADR-006
Treat "pin the exact upstream preprocessing implementation" (Prompt 02) as
possibly unachievable rather than pending. A 2026-09-27 search found no
public source repository for the Sanket checkpoint; the model card's own
GitHub link is an illustrative placeholder. `scripts/inspect_state_dict.py`
was added as the fallback path: recover architecture facts from the
checkpoint's state_dict directly instead of waiting indefinitely on upstream
code that may never surface.

## ADR-007
State-dict introspection confirmed the model parameter shapes and architecture boundaries for Sanket (`model_best.pth`), but exact MediaPipe pose feature composition and frame sampling rate remain unrecoverable without upstream source code.

Confirmed Model Contract:
| Preprocessing Fact | Value / Specification | Status |
| :--- | :--- | :--- |
| **Frames per Clip** | `8` frames | **CONFIRMED FROM WEIGHTS** (`time_embeddings` shape `(1, 8, 768)`) |
| **Input Resolution** | `224x224` RGB | **CONFIRMED FROM WEIGHTS** (`position_embeddings` `197` tokens -> $14 \times 14$ grid) |
| **Patch Size** | `16x16` pixels | **CONFIRMED FROM WEIGHTS** (`patch_embeddings.projection` kernel `(16, 16)`) |
| **Pose Vector Dimension** | `195` floats / frame | **CONFIRMED FROM WEIGHTS** (`pose_encoder.fc_in` shape `(768, 195)`) |
| **MediaPipe Landmark Set / Feature Mapping** | Exact landmark selection & ordering producing 195 dims | **UNKNOWN** (Vector length 195 is confirmed, but landmark subset/ordering cannot be determined from weights) |
| **Image Normalization Values** | Mean & Std (e.g. ImageNet `[0.45, 0.45, 0.45]`, `[0.225, 0.225, 0.225]`) | **ASSUMED FROM MODEL CARD (UNVERIFIED)** |
| **Tokenizer & Vocab Size** | `T5-base` tokenizer (`32,128` vocabulary) | **CONFIRMED FROM WEIGHTS** (`text_decoder.shared` shape `(32128, 768)`), special tokens/prompts **ASSUMED FROM MODEL CARD (UNVERIFIED)** |
| **Frame Sampling Stride / Rate** | FPS or temporal sampling policy across raw video | **UNKNOWN** (No temporal stride metadata present in checkpoint) |

Decision & Impact:
Because the 195-dim MediaPipe feature vector composition and frame sampling stride cannot be verified against a real source, any pose preprocessing pipeline constructed at this stage is explicitly classified as EXPERIMENTAL / UNVERIFIED, not a true reproduction of the Sanket training pipeline. SanketModelProvider will continue to guard against executing inference until these contracts are verified or an alternative end-to-end trained checkpoint is provided.

## ADR-008
Date: 2026-09-27
Decision: Proceed with Candidate A (23 upper-body pose + 21 left + 21 right hand 3D landmarks = 195-dim) + uniform 8-frame temporal sampling + ImageNet normalization as the EXPERIMENTAL pipeline baseline.

Status: EXPERIMENTAL / UNVERIFIED BASELINE ONLY.

Details & Assumptions:
1. Pose Feature Selection: Candidate A (23 upper-body pose landmarks x 3 + 21 left-hand landmarks x 3 + 21 right-hand landmarks x 3 = 195 floats) is implemented as the default pose landmark extraction scheme in `src/preprocessing/pose.py`.
2. Temporal Frame Sampling: Uniform sampling of 8 frames across the full video clip duration using `np.linspace(0, total_frames - 1, 8, dtype=int)` is implemented in `src/preprocessing/video.py`.
3. Image Normalization: Standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` are used.
4. Text Generation: T5 Beam Search with `num_beams=4` and `max_new_tokens=32`.

Explicit Disclaimer:
This experimental pipeline is NOT a verified reproduction of the Sanket paper/model card BLEU-1 50.6 baseline. The inference path is kept behind an explicit opt-in flag (`allow_experimental_inference=True`) in `SanketModelProvider.predict()`, and returns a prominent warning on every prediction call.

What would change if upstream code surfaces:
If the original author publishes training/preprocessing code:
- Replace `PoseLandmarkScheme` in `src/preprocessing/pose.py` with the exact landmarker set and feature ordering.
- Update temporal sampling stride/rate in `src/preprocessing/video.py`.
- Update normalization parameters if non-ImageNet values were used.
- Remove the experimental opt-in requirement once preprocessing fidelity is independently verified.


