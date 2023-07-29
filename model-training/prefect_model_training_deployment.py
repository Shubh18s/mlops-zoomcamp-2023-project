import model_training_pipeline
from prefect.deployments import Deployment
from prefect.filesystems import GCS
from prefect_gcp import GcsBucket


WORK_POOL_NAME="citibike-training-work-pool"
TRAINING_WORK_QUEUE="citibike-training-work-queue"
storage = GCS.load("citibike-training-prefect-flows") # load a pre-defined block

# print(storage)
deployment = Deployment.build_from_flow(
    flow=model_training_pipeline.model_training,
    name="citibike-model-training",
    version=2,
    work_queue_name=TRAINING_WORK_QUEUE,
    work_pool_name=WORK_POOL_NAME,
    storage=storage,
    # infra_overrides={
    #     "env": {
    #         "ENV_VAR": "value"
    #     }
    # },
)

deployment.apply()