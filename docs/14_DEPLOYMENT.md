# Deployment

## Local
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_environment.py
```

## Model
```bash
python scripts/download_sanket.py
python scripts/inspect_model_artifact.py
```

## API
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Production topology
Reverse proxy -> TLS -> API -> inference worker pool -> GPU nodes.

Use immutable containers, health checks, staged rollout, monitoring and rollback.
