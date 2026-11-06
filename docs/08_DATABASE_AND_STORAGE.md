# Database and Storage

## PostgreSQL
Store users, sessions, model versions, deployments, evaluation runs and configuration metadata.

## Object storage
Store approved model artifacts, evaluation reports and dataset manifests.

Do not use general application storage for raw signer videos.

## Model entity
id, name, version, sha256, source_uri, preprocessing_version, created_at, approval_status.

## Evaluation entity
id, model_version_id, dataset_version, metrics_json, created_at.

## Session entity
id, model_version_id, started_at, ended_at, device_class, runtime_version.
