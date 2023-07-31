import os
import pickle
import click
import mlflow
import optuna
import numpy as np
import pandas as pd
import preprocessors
import fetch_best_model_run

from dateutil.relativedelta import relativedelta
from datetime import datetime
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from sklearn.feature_extraction import DictVectorizer
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric

MODEL_NAME="sklearn-random-forest-reg-model"
MODEL_STAGE="Production"


@task(retries=2)
def read_dataframe(filename: str):
    """Reading data into a dataframe
    and cleaning the data"""
    df = pd.read_csv(filename)
    
    df.dropna(inplace=True)

    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])

    df['distance']  = df.apply(lambda row: preprocessors.get_distance_in_km(row['start_lat'],row['start_lng'],row['end_lat'],row['end_lng']),axis=1)
    df['duration'] = df.ended_at - df.started_at
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

    df['hour_of_day'] = df.started_at.dt.hour
    df['day_of_week'] = df.started_at.dt.day_of_week

    df['start_stop'] = df['start_station_id'] + '_' + df['end_station_id']
    df = df[(df.duration >= 1) & (df.duration <= 20) ].copy() #& (df.hour_of_day >= 5)

    return df

@task
def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):   
    categorical_features = ['start_stop', 'rideable_type', 'hour_of_day', 'day_of_week', 'member_casual']
    numerical_features = ['distance']
    df[categorical_features] = df[categorical_features].astype(str)
    dicts = df[categorical_features + numerical_features].to_dict(orient='records')

    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

@task
def load_preprocessor(client, run_id):
    artifact_uri = client.get_run(run_id).info.artifact_uri
    mlflow.artifacts.download_artifacts(artifact_uri = f"{artifact_uri}/preprocessor", dst_path=".")

    with open(f"preprocessor/preprocessor.b", "rb") as f_in:
        preprocessor =  pickle.load(f_in)

    return preprocessor

@task
def load_best_model(client, run_id):
    artifact_uri = client.get_run(run_id).info.artifact_uri
    logged_model = f"{artifact_uri}/model" 
    model = mlflow.pyfunc.load_model(logged_model)
    return model

@task(retries=2)
def generate_file_names(run_date: datetime.date):
    file_prefix = "JC-"
    file_suffix = "-citibike-tripdata.csv"

    ref_date = run_date - relativedelta(months=4)
    curr_date = run_date - relativedelta(months=2)

    ref_filename = f"{file_prefix}{ref_date.year:04d}{ref_date.month:02d}{file_suffix}"
    curr_filename = f"{file_prefix}{curr_date.year:04d}{curr_date.month:02d}{file_suffix}"
    
    return(ref_filename, curr_filename)


@task(retries=2)
def generate_file_path(file_name: str, raw_data_path: str ="./data/"):
    return(f"{raw_data_path}{file_name}")

@task
def create_results(df, y_pred):
    df_result = pd.DataFrame()
    df_result['start_stop'] = df['start_stop']
    df_result['rideable_type'] = df['rideable_type']
    df_result['hour_of_day'] = df['hour_of_day']
    df_result['day_of_week'] = df['day_of_week']
    df_result['member_casual'] = df['member_casual']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred

    return df_result

@flow(name="citibike model monitoring") 
def model_monitoring(run_date: datetime = None):
    TRACKING_URI="http://127.0.0.1:5000"
    client = MlflowClient(tracking_uri=TRACKING_URI)

    logger = get_run_logger()
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    logger.info("Generating reference and current data file names...")
    ref_file, curr_file = generate_file_names(run_date)

    logger.info("Generating file paths...")
    ref_file_path = generate_file_path(file_name = ref_file, raw_data_path="./data/")
    curr_file_path = generate_file_path(file_name = curr_file, raw_data_path="./data/")
    
    logger.info("Reading dataframe...")
    df_ref = read_dataframe(ref_file_path)
    df_curr = read_dataframe(curr_file_path)
    
    run_id = fetch_best_model_run.get_prod_model_run_id(client, MODEL_NAME, MODEL_STAGE)
    logger.info(f'Loading the best model with RUN_ID={run_id}...')
    model = load_best_model(client, run_id)
    preprocessor = load_preprocessor(client, run_id)

    logger.info("Preparing data ...")
    # Extract the target
    target = 'duration'
    y_ref = df_ref[target].values
    y_curr = df_curr[target].values

    ref_dicts, _ = preprocess(df_ref, preprocessor, False)
    curr_dicts, _ = preprocess(df_curr, preprocessor, False)

    logger.info(f'Applying the model..')
    ref_pred = model.predict(ref_dicts)
    curr_pred = model.predict(curr_dicts)
    ref_data = create_results(df_ref, ref_pred)
    curr_data = create_results(df_curr, curr_pred)
    cat_features = ['rideable_type', 'hour_of_day', 'day_of_week', 'member_casual']
    num_features = ['distance']
    column_mapping = ColumnMapping(
        target=None,
        prediction='predicted_duration',
        task = 'regression',
        numerical_features=num_features,
        categorical_features=cat_features
    )

    report = Report(metrics=[
        ColumnDriftMetric(column_name='predicted_duration'),
        DatasetDriftMetric()
        ]
    )
    logger.info(f'Running evidently report..')
    report.run(reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping)
    result = report.as_dict()

    #prediction drift
    print(f"Prediction drift: {result['metrics'][0]['result']['drift_score']}")
    #number of drifted columns
    print(f"Number of drifted columns: {result['metrics'][1]['result']['number_of_drifted_columns']}")
   

if __name__ == '__main__':
    """Needs datetime string as the first parameter"""
    run_date = pd.to_datetime(sys.argv[1])
    # print(run_date)
    model_monitoring(run_date)