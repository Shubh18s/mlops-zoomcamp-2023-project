from datetime import datetime
from model_scoring_pipeline import prepare_features, generate_file_names, generate_file_path
from preprocessors import get_distance_in_km
import pandas as pd

def dt(year=2023, month=3, day=15, hour=0, minute=0, second=0):
    return datetime(year, month, day, hour, minute, second) #.strftime("%m/%d/%Y %H:%M:%S")

def test_prepare_features():    
    input_data = [
        ('XYZ', 'JC019', 'JC034', None, None, 'classic_bike', 'casual', None,-74.05746818, 40.73478582, -74.05044364 ),
        ('123', 'JC019', 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,-74.05746818, 40.73478582, -74.05044364),
        ('XYZ', 'JC019', 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'electric_bike', 'member', 40.73111486,-74.05746818, 40.73478582, -74.05044364),
        ('123', 'JC019', None, "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,None, 40.73478582, -74.05044364),
        ('XYZ', 'JC019', 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,-74.05746818, 40.73478582, -74.05044364),
        ('123', None, 'JC034', "27/06/2023 16:06", "27/06/2023 16:10", 'classic_bike', 'casual', 40.73111486,-74.05746818, None, -74.05044364),     
    ]

    input_columns = ['ride_id', 'start_station_id', 'end_station_id', 'started_at', 'ended_at', 'rideable_type', 'member_casual', 'start_lat', 'start_lng', 'end_lat', 'end_lng']
    input_df = pd.DataFrame(input_data, columns=input_columns)
    result = prepare_features(input_df)
    actual_result = result.reset_index(drop=True).to_dict()

    output_data = [
        ('123', 'JC019_JC034','classic_bike', 16, 1, 'casual', 0.718987768889885, 4.0),
	    ('XYZ', 'JC019_JC034','electric_bike', 16, 1, 'member', 0.718987768889885, 4.0),
	    ('XYZ', 'JC019_JC034','classic_bike', 16, 1, 'casual', 0.718987768889885, 4.0)
    ]
    output_columns = ['ride_id', 'start_stop', 'rideable_type', 'hour_of_day', 'day_of_week', 'member_casual', 'distance', 'duration']
    expected_result = pd.DataFrame(output_data, columns=output_columns).to_dict()

    assert expected_result == actual_result

def test_preprocessor_get_distance_in_km():

    actual_result = get_distance_in_km(40.73111486, -74.05746818, 40.73478582, -74.05044364)
    expected_result = 0.718987768889885

    assert expected_result == actual_result

def test_generate_file_names():
    date = dt()
    actual_result = generate_file_names(date)

    expected_result = f"JC-{date.year:04d}{date.month:02d}-citibike-tripdata.csv"
    assert expected_result == actual_result

def test_generate_file_path():
    
    actual_result = generate_file_path("hello", "hi")

    expected_result = "hihello"
    assert expected_result == actual_result