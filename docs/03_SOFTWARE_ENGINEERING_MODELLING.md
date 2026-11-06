# Software Engineering Modelling Techniques

## Context model

```text
[Signer] -> [Camera] -> [ISL Translator] -> [English Text]
                                      \-> [Speech]
```

## Use cases
UC-01 Start translation
UC-02 Translate continuous signing
UC-03 Review uncertainty
UC-04 Enable/disable speech
UC-05 Manage approved model
UC-06 Evaluate a model
UC-07 Roll back deployment

## Sequence

```text
User -> Camera: frames
Camera -> Pipeline: frame stream
Pipeline -> FeatureExtractor: features
FeatureExtractor -> Model: temporal tensor
Model -> Decoder: prediction
Decoder -> ConfidencePolicy: candidate
ConfidencePolicy -> UI: stable result
UI -> TTS: optional speech
```

## State machine

```text
IDLE -> INITIALIZING -> READY -> CAPTURING -> PROCESSING
PROCESSING -> CONFIDENT -> CAPTURING
PROCESSING -> UNCERTAIN -> CAPTURING
```

## Data-flow levels
Level 0: Video -> Translation -> Text/Speech
Level 1: Video -> Preprocessing -> Features -> Recognition -> Translation -> Output

## Design principles
Separation of concerns, dependency inversion, immutable artifacts, explicit schemas, deterministic preprocessing, reproducibility.
