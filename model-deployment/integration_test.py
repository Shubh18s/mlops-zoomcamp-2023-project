import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import model_scoring_pipeline

def dt(year=2023, month=3, day=1, hour=0, minute=0, second=0):
    return datetime(year, month, day, hour, minute, second)

input_data = [
        ('XYZ', 'JC019', 'JC034', None, None, 'classic_bike', 'casual', None,-74.05746818, 40.73478582, -74.05044364 ),
        ('123', 'JC019', 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,-74.05746818, 40.73478582, -74.05044364),
        ('XYZ', 'JC019', 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'electric_bike', 'member', 40.73111486,-74.05746818, 40.73478582, -74.05044364),
        ('123', 'JC019', None, "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,None, 40.73478582, -74.05044364),
        ('XYZ', 'JC019', 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,-74.05746818, 40.73478582, -74.05044364),
        ('123', None, 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,-74.05746818, None, -74.05044364),     
    ]

input_columns = ['ride_id', 'start_station_id', 'end_station_id', 'started_at', 'ended_at', 'rideable_type', 'member_casual', 'start_lat', 'start_lng', 'end_lat', 'end_lng']
df_input = pd.DataFrame(input_data, columns=input_columns)


# input_file = f"{os.path.dirname(os.path.realpath('model-deployment'))}/data/2023-06.csv"
# print(input_file)


test_run_date = dt(year=2023, month=2)
# Tests will be run for 1 month ago data
test_data_date = test_run_date - relativedelta(months=1)
input_file = model_scoring_pipeline.generate_file_names(test_data_date)
input_file_path = model_scoring_pipeline.generate_file_path(file_name = input_file)

output_file = model_scoring_pipeline.generate_file_path(2022, 1)

df_input.to_csv(
    input_file_path,
    compression=None,
    index=False
)

dir_path = os.path.dirname(os.path.realpath("./model_scoring_pipeline.py"))
file_path = os.path.join(dir_path,'model_scoring_pipeline.py')
# print(dir_path)

# year = 2022
# month = 1

os.system(f'python {file_path} {test_run_date}') # Add 3rd param model_path to invoke batch.py from  here


# options = {
#         'client_kwargs': {
#             'endpoint_url': "http://localhost:4566" #S3_ENDPOINT_URL
#         }
#     }
# df = pd.read_parquet("s3://nyc-duration/in/2022-01.parquet", storage_options=options)

# print(df['predictions'].sum())