#!/bin/bash

cd ./

echo "Starting prefect server"
pipenv run prefect server start

# pipenv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api