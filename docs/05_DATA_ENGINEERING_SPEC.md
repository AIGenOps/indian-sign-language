# Data Engineering Specification

## Dataset families

### ISLTranslate
Continuous ISL-English sentence/phrase resource. The repository describes ~30k pairs and the paper describes 31k pairs.

### INCLUDE
Large-scale isolated ISL recognition resource.

### CISLR
Large-vocabulary ISL recognition resource. Verify current distribution and license before use.

## Metadata
sample_id, source_dataset, signer_id, language, sentence, gloss sequence, fps, resolution, duration, license, provenance, preprocessing_version.

## Splits
Use signer-disjoint train/validation/test splits whenever signer identity is available.

## Quality gates
Reject/quarantine corrupt videos, missing labels, duplicates, unreadable frames, invalid timestamps, ambiguous licensing, inconsistent labels.

## Augmentation
Only use meaning-preserving augmentation. Do not blindly horizontal-flip signs.

## Privacy
Raw signer video requires access control, documented purpose, retention, and deletion policy.
