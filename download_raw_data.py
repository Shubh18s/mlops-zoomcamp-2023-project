import urllib.request
import zipfile
import requests
import shutil

training_path = "./model_training/data"
deployment_path = "./model_deployment/data"

files = [('JC-202303-citibike-tripdata.csv.zip', training_path),
        ('JC-202304-citibike-tripdata.csv.zip', training_path),
        ('JC-202305-citibike-tripdata.csv.zip', training_path),
        ('JC-202306-citibike-tripdata.csv.zip', deployment_path)]

print("Downloading files...")
for (file, path) in files:
    url=f"https://s3.amazonaws.com/tripdata/{file}"
    zip_path, _ = urllib.request.urlretrieve(url)
    save_path=f"{path}"
    
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(save_path)
    print(f"{file} successfully downloaded in {save_path}")

print("Download successfull.")