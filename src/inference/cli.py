import argparse
from src.config import Settings
from src.models.sanket import SanketModelProvider

def main() -> None:
    parser = argparse.ArgumentParser(description="ISL Translator inference entrypoint")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    settings = Settings()
    checkpoint = args.checkpoint or settings.model_checkpoint
    provider = SanketModelProvider(checkpoint)
    provider.load()

    print("Checkpoint exists and passed the adapter load gate.")
    print("Inference remains gated until exact upstream preprocessing is pinned.")
    print("See docs/20_PROMPTS.md Prompt 02.")

if __name__ == "__main__":
    main()
