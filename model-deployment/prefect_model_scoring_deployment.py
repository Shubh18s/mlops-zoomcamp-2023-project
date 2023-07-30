import model_scoring_pipeline
from prefect.deployments import Deployment
from prefect.filesystems import GCS
from prefect_gcp import GcsBucket


WORK_POOL_NAME="citibike-scoring-work-pool"
SCORING_WORK_QUEUE="citibike-scoring-work-queue"
storage = GCS.load("citibike-scoring-prefect-flows") # load a pre-defined block

# print(storage)
deployment = Deployment.build_from_flow(
    flow=model_scoring_pipeline.apply_model,
    name="citibike-model-scoring",
    version=2,
    work_queue_name=SCORING_WORK_QUEUE,
    work_pool_name=WORK_POOL_NAME,
    storage=storage,
    # infra_overrides={
    #     "env": {
    #         "ENV_VAR": "value"
    #     }
    # },
)

deployment.apply()