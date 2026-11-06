# Threat Model

## Spoofing
Unauthorized inference API access.
Mitigation: authentication, short-lived credentials, rate limits.

## Tampering
Model or preprocessing modification.
Mitigation: SHA-256 verification and signed release artifacts.

## Repudiation
Missing traceability.
Mitigation: non-sensitive audit metadata.

## Information disclosure
Signer video in logs.
Mitigation: frame logging disabled.

## Denial of service
Unbounded frame requests.
Mitigation: quotas, bounded buffers, rate limits.

## Elevation of privilege
Inference user accesses model administration.
Mitigation: RBAC and separate management plane.
