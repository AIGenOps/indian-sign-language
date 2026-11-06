# Annotation Specification

## Canonical JSONL

```json
{
  "sample_id": "isl_000001",
  "video_uri": "data://...",
  "language": "isl",
  "signer_id": "signer_001",
  "sentence_en": "I need water",
  "glosses": [
    {"gloss": "I", "start_ms": 0, "end_ms": 700},
    {"gloss": "NEED", "start_ms": 700, "end_ms": 1500},
    {"gloss": "WATER", "start_ms": 1500, "end_ms": 2500}
  ],
  "quality": {"video_ok": true, "annotation_ok": true},
  "provenance": {"source": "owned_dataset", "license": "documented"}
}
```

## States
UNREVIEWED -> FIRST_PASS -> SECOND_PASS -> SIGNER_VALIDATED -> APPROVED / REJECTED.

Use independent annotation on a representative subset and adjudicate disagreements.

Every export receives an annotation version and checksum.
