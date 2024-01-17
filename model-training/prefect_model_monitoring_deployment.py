import model_monitoring_pipeline
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.filesystems import GCS


WORK_POOL_NAME="citibike-monitoring-work-pool"
MONITORING_WORK_QUEUE="citibike-monitoring-work-queue"
storage = GCS.load("citibike-training-prefect-flows") # load a pre-defined block

deployment = Deployment.build_from_flow(
    flow=model_monitoring_pipeline.model_monitoring,
    name="citibike-model-monitoring",
    version=2,
    work_queue_name=MONITORING_WORK_QUEUE,
    work_pool_name=WORK_POOL_NAME,
    storage=storage,
    schedule=(CronSchedule(cron="30 12 15 * *", timezone="Australia/Melbourne"))
)

deployment.apply()