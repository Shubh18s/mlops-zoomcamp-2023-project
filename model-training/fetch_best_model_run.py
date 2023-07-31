import mlflow.pyfunc
from mlflow.tracking import MlflowClient

TRACKING_URI="http://127.0.0.1:5000"

def get_prod_model_run_id(client: MlflowClient, model_name: str, stage: str="Production"):
    stages_list = []
    stages_list.append(stage)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    latest_versions = client.get_latest_versions(
        name=model_name,
        stages=stages_list)[0]
    return(latest_versions.run_id)

# def download_model_artifacts(model_name: str, stage: str="Production"):
#     client = MlflowClient(tracking_uri=TRACKING_URI)
#     artifact_uri = client.get_run(get_prod_model_run_id(model_name, stage)).info.artifact_uri
#     print(artifact_uri)
    # 
    # print(mlflow.artifacts.download_artifacts(run_id = get_best_run_source(model_name, stage), dst_path= "./artifacts"))

if __name__ == "__main__":
    get_prod_model_run_id("sklearn-random-forest-reg-model", "Production")