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

    df = df[(df.duration >= 1) & (df.duration <= 20) ].copy() #& (df.hour_of_day >= 5)

    return df

@task
def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    categorical = ['start_station_id', 'end_station_id', 'rideable_type', 'hour_of_day', 'day_of_week', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_casual']
    numerical = ['distance']
    df[categorical] = df[categorical].astype(str)
    df['start_stop'] = df['start_station_id'] + '_' + df['end_station_id']
    categorical = ['start_stop', 'rideable_type', 'hour_of_day', 'day_of_week', 'member_casual'] # 'start_station_id', 'end_station_id 'member_casual', 'hour_of_day', 'day_of_week', 'start_station_id', 'end_station_id' 'start_station_name', 'end_station_name', 'start_lat_lng', 'end_lat_lng'
    numerical = ['distance']

    dicts = df[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

@task
def generate_file_names(score_date: datetime.date):
    file_prefix = "JC-"
    file_suffix = "-citibike-tripdata.csv"
    score_filename = f"{file_prefix}{score_date.year:04d}{score_date.month:02d}{file_suffix}"
    return(score_filename)

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

@task
def generate_file_path(file_name: str, raw_data_path: str ="./data/"):
    return(f"{raw_data_path}{file_name}")

@task
def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['started_at'] = df['started_at']
    df_result['start_station_id'] = df['start_station_id']
    df_result['end_station_id'] = df['end_station_id']
    df_result['rideable_type'] = df['rideable_type']
    df_result['hour_of_day'] = df['hour_of_day']
    df_result['day_of_week'] = df['day_of_week']
    df_result['member_casual'] = df['member_casual']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    df_result['duration_deviation'] = df_result['actual_duration'] - df_result['predicted_duration']
    df_result['model_version'] = run_id

    df_result.to_csv(output_file, index=False)
    return output_file

@flow(name="citibike data scoring") 
def apply_model(run_date: datetime = None):
    TRACKING_URI="http://127.0.0.1:5000"
    client = MlflowClient(tracking_uri=TRACKING_URI)

    logger = get_run_logger()
    run_date = datetime.today()
    score_date = run_date - relativedelta(months=1)
    
    logger.info("Generating file names...")
    input_file = generate_file_names(score_date)

    logger.info("Generating file paths...")
    input_file_path = generate_file_path(file_name = input_file, raw_data_path="./data/")
    output_file = f'gs://citibike-deployment-scoring-artifacts/output/{score_date.year:04d}-{score_date.month:02d}.csv'
    
    run_id = fetch_best_model_run.get_prod_model_run_id(client, MODEL_NAME, MODEL_STAGE)
    logger.info(f'Loading the best model with RUN_ID={run_id}...')
    model = load_best_model(client, run_id)
    preprocessor = load_preprocessor(client, run_id)

    logger.info("Preparing score data ...")
    df_score = read_dataframe(input_file_path)
    dicts, _ = preprocess(df_score, preprocessor, False)

    logger.info(f'applying the model..')
    y_pred = model.predict(dicts)
    
    logger.info(f'saving the result to {output_file}..')
    save_results(df_score, y_pred, run_id, output_file)
    return output_file

def run():
    # run_date = datetime.today()
    apply_model()
    

if __name__ == '__main__':
    run()