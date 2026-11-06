# First-Day Execution

1. Create environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install:
```bash
pip install -r requirements.txt
```

3. Verify:
```bash
python scripts/verify_environment.py
```

4. Download:
```bash
python scripts/download_sanket.py
```

5. Inspect/hash:
```bash
python scripts/inspect_model_artifact.py
```

6. Test:
```bash
pytest
```

7. Execute Prompt 02 from docs/20_PROMPTS.md.

8. Reproduce a deterministic sample before enabling live camera inference.

9. Record model hash, source revision, preprocessing revision, Python/PyTorch/CUDA/GPU, latency and output.

10. Only then enable streaming.
