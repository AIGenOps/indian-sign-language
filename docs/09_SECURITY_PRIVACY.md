# Security and Privacy

## Assets
Signer video, model weights, datasets, session data, API credentials and evaluation artifacts.

## Threats
Unauthorized camera access, raw-video leakage, malicious model artifact, dependency compromise, API abuse, model theft and insecure logs.

## Controls
- explicit camera permission and active-camera indicator
- no raw-frame logging by default
- SHA-256 model verification
- dependency pinning/scanning
- TLS outside localhost
- authentication/RBAC
- rate limits and request validation
- audit logs without raw media
- constrained language decoding

## Privacy
Prefer local processing. Cloud processing requires documented purpose, notice/consent, retention and deletion controls.
