#!/bin/bash

ARTIFACT_ROOT="gs://citibike-mlflow-artifacts/"
echo "Starting mlflow server"
pipenv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root $ARTIFACT_ROOT --artifacts-destination $ARTIFACT_ROOT