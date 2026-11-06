# Operations Runbook

## Model failure
Check artifact existence, SHA-256, PyTorch/CUDA compatibility and model files. Run inspect_model_artifact.py. Review logs.

## Camera failure
Check permission, device index, resolution, FPS and OpenCV capture.

## Latency increase
Inspect GPU/CPU, frame queue, preprocessing, model inference and decoder time.

## Incorrect output
Record model version, preprocessing version, signer cohort, environment, confidence and prediction. Do not store raw video unless authorized.
