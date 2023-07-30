#!/bin/bash

WORK_POOL_NAME="citibike-scoring-work-pool"
SCORING_WORK_QUEUE="citibike-scoring-work-queue"
# echo "$WORK_POOL_NAME"
echo "Creating work-pool $WORK_POOL_NAME ..."
pipenv run prefect work-pool create $WORK_POOL_NAME --type process

echo "Creating work queue $SCORING_WORK_QUEUE for $WORK_POOL_NAME ..."
pipenv run prefect work-queue create $SCORING_WORK_QUEUE --pool $WORK_POOL_NAME

echo "Starting agent for $WORK_POOL_NAME ..."
pipenv run prefect agent start --pool $WORK_POOL_NAME --work-queue $SCORING_WORK_QUEUE

# prefect deployment run 'main-flow-hw2/taxi_gcs_2023_data_hw2