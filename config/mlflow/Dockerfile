# FROM python:3.10-slim
# # Install python packages
# RUN pip install cryptography
# RUN pip install mlflow pymysql MLFLOW_S3_ENDPOINT_URL=http://storage:9000

# FROM python:3.8-slim-buster
# WORKDIR /usr/src/app
# RUN pip3 install -y cryptography mlflow psycopg2-binary pymysql google-cloud-storage
# ENV 
# CMD mlflow server \
#     --backend-store-uri mysql+pymysql://MLFLOW:temporal@db:3306/DBMLFLOW \
#     --default-artifact-root gs://citibike-mlops/mlflow/ \
#     --host 0.0.0.0

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

# RUN gcloud auth activate-service-account --key-file ${GOOGLE_APPLICATION_CREDENTIALS}