install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	ruff check src scripts tests

format:
	ruff format src scripts tests

test:
	pytest

verify:
	python scripts/verify_environment.py

download-model:
	python scripts/download_sanket.py

inspect-model:
	python scripts/inspect_model_artifact.py

inspect-state-dict:
	python scripts/inspect_state_dict.py

api:
	uvicorn src.api.main:app --host 127.0.0.1 --port 8000
