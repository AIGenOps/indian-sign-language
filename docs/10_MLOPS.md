# MLOps Specification

## Experiment record
Record git commit, dataset version, model version, configuration, hardware, software, seed, metrics and artifact checksums.

## Model states

```text
CANDIDATE -> VALIDATED -> STAGING -> SHADOW -> PRODUCTION -> DEPRECATED
```

## Promotion gates
Reproducible evaluation, signer-disjoint test, robustness test, latency test, security review and license/provenance review.

## Rollback
Retain the previous approved model and a deployment manifest that can restore it.
