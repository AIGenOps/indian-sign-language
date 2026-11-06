# API Specification

## GET /health
Returns service/model status.

## GET /ready
Returns readiness state.

## POST /v1/sessions
Creates an inference session.

## POST /v1/sessions/{session_id}/frames
Accepts a bounded image frame and timestamp.

Example translation event:

```json
{
  "status": "confirmed",
  "text": "I need water",
  "confidence": 0.91,
  "model_version": "sanket-baseline"
}
```

Uncertain:

```json
{
  "status": "uncertain",
  "text": null,
  "confidence": 0.42,
  "reason": "below_threshold"
}
```

Remote deployments require TLS, authentication, authorization, rate limiting, request-size limits and no raw-frame logging.
