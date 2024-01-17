import urllib.request
import zipfile
import requests
import shutil
from datetime import datetime
from dateutil.relativedelta import relativedelta

currentDay = datetime.today()

train_date = currentDay - relativedelta(months=5)
val_date = currentDay - relativedelta(months=4)
test_date = currentDay - relativedelta(months=3)
predict_date = currentDay - relativedelta(months=2)


training_path = "./model-training/data"
deployment_path = "./model-deployment/data"

files = [(f'JC-{train_date.year:04d}{train_date.month:02d}-citibike-tripdata.csv.zip', training_path),
        (f'JC-{val_date.year:04d}{val_date.month:02d}-citibike-tripdata.csv.zip', training_path),
        (f'JC-{test_date.year:04d}{test_date.month:02d}-citibike-tripdata.csv.zip', training_path),
        (f'JC-{predict_date.year:04d}{predict_date.month:02d}-citibike-tripdata.csv.zip', training_path),
        (f'JC-{predict_date.year:04d}{predict_date.month:02d}-citibike-tripdata.csv.zip', deployment_path)]

print("Downloading files...")
for (file, path) in files:
    url=f"https://s3.amazonaws.com/tripdata/{file}"
    zip_path, _ = urllib.request.urlretrieve(url)
    save_path=f"{path}"
    
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(save_path)
    print(f"{file} successfully downloaded in {save_path}")
shutil.rmtree(f"{training_path}/__MACOSX")
shutil.rmtree(f"{deployment_path}/__MACOSX")
print("Download successfull.")