import importlib
import platform
import sys

PACKAGES = [
    "torch", "torchvision", "transformers", "cv2", "mediapipe",
    "numpy", "pydantic", "fastapi", "huggingface_hub"
]

print("Python:", sys.version)
print("Platform:", platform.platform())

failures = 0

for name in PACKAGES:
    try:
        module = importlib.import_module(name)
        print(f"{name}: OK", getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(f"{name}: FAILED - {exc}")
        failures += 1

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
except Exception as exc:
    print("Torch diagnostic failed:", exc)
    failures += 1

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        print("mediapipe.solutions: available (legacy Holistic/Pose/Hands API present)")
    else:
        print(
            "mediapipe.solutions: MISSING. This mediapipe build only exposes the "
            "newer Tasks API. Reproducing the Sanket pose-feature pipeline needs "
            "the legacy Solutions API -- see requirements.txt and "
            "MODEL_DOWNLOAD_STATUS.md before proceeding."
        )
        failures += 1
except Exception as exc:
    print("mediapipe.solutions diagnostic failed:", exc)
    failures += 1

if failures > 0:
    print(f"\nEnvironment verification completed with {failures} failure(s).")
    sys.exit(1)
else:
    print("\nEnvironment verification passed successfully!")
    sys.exit(0)

