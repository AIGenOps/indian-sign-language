# Production Acceptance Test Plan

## Quality
Required: WER, BLEU-1/2/3/4, ROUGE-L, signer-disjoint results, per-condition results and confidence calibration.

## Latency
Engineering target:
- p50 <= 250 ms
- p95 <= 500 ms

Final threshold must be validated against the approved hardware profile.

## Reliability
- 4-hour continuous test
- camera reconnect
- model reload
- GPU recovery
- no unbounded memory growth

## Safety
- no fabricated text on failure
- uncertainty below threshold
- raw frames absent from logs
- model checksum verified

Production status requires all mandatory SRS requirements to map to tests.
