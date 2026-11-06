# Test Strategy

## Unit
Preprocessing, landmark normalization, temporal buffering, confidence policy, stabilization, schemas and artifact validation.

## Integration
Camera -> preprocessing -> model -> decoder -> UI -> TTS.

## ML
Known-sample regression, unseen signer, lighting, distance, background, signing speed, sentence length and OOD tests.

## Performance
Cold/warm start, p50/p95/p99 latency, FPS, GPU memory, CPU, memory growth and long-session stability.

## Security
Dependency scan, secret scan, malformed input, upload limits, auth bypass, path traversal and model tampering.

A successful demo is not a release.
