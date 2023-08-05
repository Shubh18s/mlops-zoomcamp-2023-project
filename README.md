## NYC citibike batch prediction 
The project is intended towards predicting the ride durations for NYC citibike riders in order to understand the deviation from the actual duration that could eventually lead to analysing the infrastructure needs and any overhauls required over time.


### Run Setup
1. Create a GCP vm like shown in the lecture videos
2. Create 4 buckets in Google Cloud Storage namely - 
    ["citibike-mlflow-artifacts", 
    "citibike-deployment-scoring-artifacts", 
    "citibike-training-prefect-flows", 
    "citibike-scoring-prefect-flows"]
3. Make sure that the vm created in step 1 has access to these buckets
4. Clone the repository
5. Navigate to the project folder and run "python download_raw_data.py".
6. Open a new terminal and run "pip install pipenv"
7. Open a new terminal and navigate to "model-training" folder and run "pipenv shell" and then run "pipenv install"
8. Open a new terminal and navigate to "model-deployment" folder and run "pipenv shell" and then run "pipenv install"
9. Open a new terminal and run "gcloud auth application-default login"

### Running mlflow and prefect server - 
1. Open a new terminal and navigate to "model-training" folder and run "bash run_mlflow_server.sh"
2. Open a new terminal and navigate to "model-training" folder and run "bash run_prefect_server.sh"
3. Open a new terminal and navigate to "model-training" folder and run "pipenv shell" and then run "prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api"

### Creating agents for Model training, monitoring and deployment - 
1. Open a new terminal
    Navigate to "model-training" folder and run "bash run_model_training_agent.sh"
2. Open a new terminal
    Navigate to "model-training" folder and run "bash run_model_monitoring_agent.sh"
    
### How to run Training - 
1. Open a new terminal
    Navigate to "model-training" folder and run "bash run_deploy_model_training_monitoring_pipeline.sh"

### How to run Monitoring - 
1. Open a new terminal
    Navigate to "model-training" folder and run "bash run_deploy_model_training_monitoring_pipeline.sh"

### How to run Scoring - 
1. Open a new terminal
    Navigate to "model-deployment" folder and run "bash run_model_scoring_agent.sh"
2. Open a new terminal
    Navigate to "model-deployment" folder and run "bash run_deploy_model_scoring_pipeline.sh.sh"

### How to run tests - 
#### Unit Tests - 
1. Open a new terminal, navigate to "model-deployment" folder and run "pipenv shell"
2. Navigate to "tests/" folder and run "pytest"

#### Integration Test - 
1. Open a new terminal, navigate to "model-deployment" folder and run "pipenv shell"
2. run "python integration_test.py"