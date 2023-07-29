#!/bin/bash

# pwd
echo "Registering GCS bucket block..."
pipenv run python create_gcs_bucket_block.py

# pipenv run prefect block register -m prefect-gcp
echo "Building model training deployment..."
# pipenv run prefect deployment build model_training_pipeline.py:model_training -n citibike-model-training --storage-block gcs-bucket/citibike-training-prefect-flows/model-training
pipenv run python prefect_model_training_deployment.py