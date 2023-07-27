import urllib.request
import zipfile
import requests
import shutil

files = [('JC-202304-citibike-tripdata.csv.zip', './data'),
        ('JC-202305-citibike-tripdata.csv.zip', './data'),
        ('JC-202306-citibike-tripdata.csv.zip', './data')]

print("Downloading files...")
for (file, path) in files:
    url=f"https://s3.amazonaws.com/tripdata/{file}"
    zip_path, _ = urllib.request.urlretrieve(url)
    save_path=f"{path}"
    
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(save_path)
    print(f"{file} successfully downloaded in {save_path}")
shutil.rmtree(f"{save_path}/__MACOSX")
print("Download successfull.")