from pathlib import Path
import hashlib
import sys

path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/model_best.pth")

if not path.exists():
    raise SystemExit(f"Missing model artifact: {path}")

h = hashlib.sha256()
size = 0
with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        size += len(chunk)
        h.update(chunk)

print("Artifact:", path)
print("Bytes:", size)
print("SHA256:", h.hexdigest())
print("WARNING: verify provenance before production use.")
print()
print("For layer-level inspection (parameter names/shapes, not just the file hash),")
print("run: python scripts/inspect_state_dict.py", path)
