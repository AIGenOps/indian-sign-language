# ML System Design

## Initial baseline: Sanket

Reported architecture:
- TimeSformer visual encoder
- 2-layer Transformer pose/skeleton encoder
- gated multimodal fusion
- T5-base language decoder
- INCLUDE training data

## Production interface

```python
class SignLanguageTranslator:
    def load(self, artifact_path: str) -> None: ...
    def preprocess(self, frames): ...
    def predict(self, batch): ...
    def postprocess(self, prediction): ...
```

## Model lifecycle
Acquire -> verify -> benchmark -> validate -> register -> stage -> shadow/canary -> promote -> monitor -> rollback.

## Alternative
Benchmark a pose-first ST-GCN + Transformer + CTC continuous recognition pipeline, then compare against Sanket using signer-disjoint evaluation.

## OOD policy
If confidence is below threshold or input quality is unacceptable:

```json
{"status":"uncertain","text":null,"reason":"low_confidence"}
```

Never generate unsupported text.
