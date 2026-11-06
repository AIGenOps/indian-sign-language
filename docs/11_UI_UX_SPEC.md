# UI/UX Specification

## Primary screen
- camera preview
- live translation
- confirmed translation
- confidence/uncertainty state
- speech toggle
- pause/resume
- settings
- privacy indicator
- model status

## Flow
Open -> permission -> Ready -> signing detected -> translation -> stabilized sentence -> optional speech.

## Accessibility
Large readable text, keyboard navigation, screen-reader labels, high contrast, and no critical state conveyed only by color.

## Error handling
Use human-readable status messages and do not expose internal stack traces to users.
