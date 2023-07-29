from mlflow.tracking import MlflowClient

client = MlflowClient()

experiments = (
    client.search_experiments()
)  # returns a list of mlflow.entities.Experiment

run = client.create_run(experiments[0].experiment_id)  # returns mlflow.entities.Run
client.log_param(run.info.run_id, "hello", "world")
client.set_terminated(run.info.run_id)
