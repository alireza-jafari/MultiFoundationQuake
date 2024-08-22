
import os
import matplotlib
import pandas as pd
import requests
from io import StringIO

from datetime import datetime, timedelta
matplotlib.use('TkAgg')



def fetch_earthquake_data(start_time, end_time, min_lat, max_lat, min_lon, max_lon, min_magnitude):
    # Initialize parameters for the loop
    current_start_time = start_time
    all_data = []

    while current_start_time < end_time:
        print(current_start_time)
        # Define the chunk's end time
        chunk_end_time = current_start_time + timedelta(days=20*365)  # Adjust the timedelta as needed

        # Ensure the chunk end time does not exceed the overall end time
        if chunk_end_time > end_time:
            chunk_end_time = end_time

        # Construct the query URL
        url = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
            f"?starttime={current_start_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            f"&endtime={chunk_end_time.strftime('%Y-%m-%dT%H:%M:%S')}"
            f"&minlatitude={min_lat}"
            f"&maxlatitude={max_lat}"
            f"&minlongitude={min_lon}"
            f"&maxlongitude={max_lon}"
            f"&minmagnitude={min_magnitude}&eventtype=earthquake&orderby=time"
        )

        # Make the HTTP request
        response = requests.get(url)

        if response.status_code == 200:
            data = response.content.decode('utf-8')
            df = pd.read_csv(StringIO(data))
            all_data.append(df)
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)

        # Update the start time for the next loop iteration 976,394   87,773   13,985  775
        current_start_time = chunk_end_time

    # Combine all DataFrames into a single DataFrame
    full_data = pd.concat(all_data, ignore_index=True)
    return full_data


# Define the geographical bounds
start_time = datetime(1983, 10, 1)
end_time = datetime(2023, 10, 1)
Center_Lat = 34
Center_Long = -117
min_lat, max_lat  = (Center_Lat -2), (Center_Lat +2)
min_lon, max_lon = (Center_Long -3), (Center_Long +3)
# min_lat, max_lat, min_lon, max_lon = -90, 90, -180, 180
min_magnitude = 4.5


df = fetch_earthquake_data(start_time, end_time, min_lat, max_lat, min_lon, max_lon, min_magnitude)

# df.to_pickle("new_data.pkl")
