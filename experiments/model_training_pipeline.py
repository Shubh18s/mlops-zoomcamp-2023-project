import os
import pickle
import click
import mlflow
import optuna
import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from datetime import datetime
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from math import radians, sin, cos, acos, pi, atan2, sqrt
from sklearn.feature_extraction import DictVectorizer

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from optuna.samplers import TPESampler

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def deg2rad(deg):
  return (deg * (pi/180))
  
def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    R = 6371 # Radius of the earth in km 3963 for miles 6371 for kms
    dLat = deg2rad(lat2-lat1) # deg2rad below
    dLon = deg2rad(lon2-lon1) 
    a = sin(dLat/2) * sin(dLat/2) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dLon/2) * sin(dLon/2)
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    # d = haversine((lat1, lon1), (lat2, lon2), unit='km')
    return d

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task(retries=2)
def read_dataframe(filename: str):
    """Reading data into a dataframe
    and cleaning the data"""
    df = pd.read_csv(filename)
    
    df.dropna(inplace=True)

    df['started_at'] = pd.to_datetime(df['started_at'])
    df['ended_at'] = pd.to_datetime(df['ended_at'])

    df['distance']  = df.apply(lambda row: getDistanceFromLatLonInKm(row['start_lat'],row['start_lng'],row['end_lat'],row['end_lng']),axis=1)
    df['duration'] = df.ended_at - df.started_at
    df.duration = df.duration.apply(lambda td: td.total_seconds()/60)

    df['hour_of_day'] = df.started_at.dt.hour
    df['day_of_week'] = df.started_at.dt.day_of_week

    df = df[(df.duration >= 1) & (df.duration <= 20) ].copy() #& (df.hour_of_day >= 5)

    return df

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

@task(retries=2, name="read and clean data and create dicts")
def run_data_prep(df_train, df_val, df_test):
    
    # Extract the target
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Fit the DictVectorizer and preprocess data
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    return(X_train, y_train, X_val, y_val, X_test, y_test)
    # Create dest_path folder unless it already exists
    # os.makedirs(dest_path, exist_ok=True)

    # # Save DictVectorizer and datasets
    # logging.info(f"Saving pickle in {dest_path}")
    # dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    # dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    # dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    # dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

@task(retries=1, name="run optimization")
def run_optimization(X_train, y_train, X_val, y_val, num_trials: int = 10, experiment_name: str = HPO_EXPERIMENT_NAME):

    mlflow.set_experiment(experiment_name)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
            'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
            'random_state': 42,
            'n_jobs': -1
        }

        with mlflow.start_run():
            mlflow.set_tag("model", "RandomForestRegresor")
            mlflow.log_params(params)
            
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(rf, artifact_path="models")

        return rmse

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

def train_and_log_model(X_train, y_train, X_test, y_test, params, experiment_name: str):
    RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():

        required_params = {}
        for param in RF_PARAMS:
            required_params[param] = int(params[param])
        
        rf = RandomForestRegressor(**required_params)
        rf.fit(X_train, y_train)

        # Evaluate model on the test set
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

@task(retries=2, name="run and register model")
def run_register_model(X_train, y_train, X_test, y_test, logger, top_n: int = 5):

    client = MlflowClient()

    logger.info(f"Retrieving the top {top_n} models for {HPO_EXPERIMENT_NAME}")
    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    
    for run in runs:
        train_and_log_model(X_train, y_train, X_test, y_test, params=run.data.params, experiment_name = EXPERIMENT_NAME)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs( 
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"])[0]

    # Register the best model
    best_run_id = best_run.info.run_id
    mlflow.register_model(model_uri=f"runs:/{best_run_id}/model", name="RandomForestRegressorStage")

@task
def generate_file_names(run_date: datetime.date):
    file_prefix = "JC-"
    file_suffix = "-citibike-tripdata.csv"

    train_date = run_date - relativedelta(months=3)
    val_date = run_date - relativedelta(months=2)
    test_date = run_date - relativedelta(months=1)

    train_filename = f"{file_prefix}{train_date.year:04d}{train_date.month:02d}{file_suffix}"
    val_filename = f"{file_prefix}{val_date.year:04d}{val_date.month:02d}{file_suffix}"
    test_filename = f"{file_prefix}{test_date.year:04d}{test_date.month:02d}{file_suffix}"
    
    return(train_filename, val_filename, test_filename)


@task
def generate_file_path(file_name: str, raw_data_path: str ="./data/"):
    return(f"{raw_data_path}{file_name}")

@flow(name="main flow") 
def model_training(run_date):
    logger = get_run_logger()
    logger.info("Generating file names...")
    train_file, val_file, test_file = generate_file_names(run_date)

    logger.info("Generating file paths...")
    train_file_path = generate_file_path(file_name = train_file, raw_data_path="./data/")
    val_file_path = generate_file_path(file_name = val_file, raw_data_path="./data/")
    test_file_path = generate_file_path(file_name = test_file, raw_data_path="./data/")
    
    logger.info("Reading dataframe...")
    df_train = read_dataframe(train_file_path)
    df_val = read_dataframe(val_file_path)
    df_test = read_dataframe(test_file_path)

    logger.info("Running data prep...")
    X_train, y_train, X_val, y_val, X_test, y_test = run_data_prep(df_train, df_val, df_test)

    logger.info("Running parameter optimization using Random Forest Regressor...")
    run_optimization(X_train, y_train, X_val, y_val, num_trials=10, experiment_name=HPO_EXPERIMENT_NAME)

    logger.info("Registering model with best params...")
    run_register_model(X_train, y_train, X_test, y_test, logger, top_n=5)

def run():
    # year = int(sys.argv[2]) #2022
    # month = int(sys.argv[3]) #2
    run_date = datetime.today()
    model_training(run_date)
    

if __name__ == '__main__':
    run()