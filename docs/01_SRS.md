# Software Requirements Specification

## Functional requirements

FR-001: Support configurable camera/video input.

FR-002: Normalize frames according to the active model contract.

FR-003: Detect/track the signer and reject unusable frames.

FR-004: Support hand, body and facial features where required.

FR-005: Maintain a bounded temporal window.

FR-006: Recognize continuous signing without manually cropped signs.

FR-007: Translate signing into English.

FR-008: Attach confidence or explicit uncertainty to every emitted result.

FR-009: Stabilize streaming predictions and suppress duplicate partial outputs.

FR-010: Display current and confirmed translations separately.

FR-011: Provide optional text-to-speech.

FR-012: Prefer local processing and avoid raw-frame persistence by default.

FR-013: Model failures must produce explicit errors, never fabricated text.

FR-014: Record exact model and preprocessing versions.

FR-015: Maintain non-sensitive audit/telemetry metadata.

## Non-functional requirements

NFR-001: Target visible translation p95 latency <= 500 ms on the approved hardware profile.

NFR-002: Target service availability >= 99.5% for deployed service components.

NFR-003: No silent inference failures.

NFR-004: No production secrets in source control.

NFR-005: No raw camera logging without documented consent and retention policy.

NFR-006: Keyboard navigation, readable text, and high contrast.

NFR-007: Reproducible evaluation using versioned configuration and artifacts.

NFR-008: Model/provider abstraction.

NFR-009: Horizontal scalability for cloud inference where enabled.

NFR-010: Health, readiness, structured logs and metrics.

## Acceptance
Acceptance criteria are defined in `docs/15_ACCEPTANCE_TEST_PLAN.md`.
