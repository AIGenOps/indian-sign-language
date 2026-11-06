# Software Architecture Document

## High-level architecture

```text
Camera
  |
Capture / Preprocess
  |
Hand + Pose + Face Features
  |
Temporal Sign Recognition
  |
Gloss/Semantic Representation
  |
Language Translation
  |
Confidence + Stabilization
  +------> Text
  +------> TTS
```

## Components
1. CaptureAdapter
2. Preprocessor
3. FeatureExtractor
4. ModelProvider
5. TemporalDecoder
6. TranslationDecoder
7. ConfidencePolicy
8. SessionManager
9. TTSAdapter
10. API
11. Observability
12. ModelRegistry

## Deployment
Edge-first is preferred. Hybrid/cloud deployment is supported only with explicit privacy controls.

## Failure boundaries
Camera failure, invalid frame, signer absent, feature extraction failure, GPU failure, model load failure, low confidence, API failure, TTS failure.

Every boundary has an explicit state; no failure becomes invented text.

## Architectural rule
Application code depends on `ModelProvider`, not on Sanket internals. This allows future model replacement without rewriting the application.
