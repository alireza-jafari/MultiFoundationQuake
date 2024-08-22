# Re-importing necessary libraries and reloading the data due to code execution state reset
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('TkAgg')  # You can also try other backends like 'Agg', 'Qt5Agg', etc.
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import requests
from io import StringIO
import json
import random
import numpy as np
import itertools
import math
import torch
import os
from tqdm import tqdm
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt




#-------------------------------------------------------------
seed = 100
# Set the random seed for Python's random module
random.seed(seed)
# Set the random seed for NumPy
np.random.seed(seed)
print('seed :',seed)
#-------------------------------------------------------------

df = pd.read_pickle('earthquake_data.pkl')
start_time = "1950-01-01"
end_time = "2024-05-01"

min_magnitude = 0.0
# Define the geographical bounds
min_lat, max_lat, min_lon, max_lon = 32, 36, -120, -114
df = df[df['longitude'] >= min_lon]
df = df[df['longitude'] <= max_lon]
df = df[df['latitude'] >= min_lat]
df = df[df['latitude'] <= max_lat]
df.reset_index(drop=True, inplace=True)
#-------------------------------------------------------------




# Define the ranges and step
step = 0.1

# Create bins
num_steps_lat = int((max_lat - min_lat) / step)
lat_bins = pd.IntervalIndex.from_breaks(np.linspace(min_lat, max_lat, num_steps_lat + 1))
num_steps_lon = int((max_lon - min_lon) / step)
lon_bins = pd.IntervalIndex.from_breaks(np.linspace(min_lon, max_lon, num_steps_lon + 1))

# Generate all combinations of bins
combinations = list(itertools.product(lat_bins, lon_bins))

# Assign a unique number to each combination
combination_mapping = {combo: i+1 for i, combo in enumerate(combinations)}

# Assign each data point to a bin
df['lat_bin'] = pd.cut(df['latitude'], bins=lat_bins, include_lowest=True)
df['lon_bin'] = pd.cut(df['longitude'], bins=lon_bins, include_lowest=True)

# Remove rows with NaN values in 'lat_bin' or 'lon_bin'
df = df.dropna(subset=['lat_bin', 'lon_bin'])

# Map each row in the DataFrame to its unique number, handling missing keys
df['square'] = df.apply(lambda row: combination_mapping.get((row['lat_bin'], row['lon_bin']), -1), axis=1)

# List of unique squares that contain data
unique_values = df['square'].unique()
list_of_squares = np.sort(unique_values)

print(df)




# Convert to pandas Timestamps
timestamps = pd.to_datetime(df['time'], utc=True)
# Convert to naive timestamps (remove timezone)
# Convert to a Series
timestamp_series = pd.Series(timestamps)
# Convert to naive timestamps (remove timezone) using dt accessor
naive_timestamp_series = timestamp_series.dt.tz_convert(None)
# Format the timestamps to remove fractional seconds
formatted_timestamps = naive_timestamp_series.dt.strftime('%Y-%m-%d %H:%M:%S')
# Convert the index to datetime
df.index = pd.to_datetime(formatted_timestamps)
print(df)


time_interval = 14 #fortnightly
# Define the start and end dates
start_date = pd.to_datetime("1950-01-03")
end_date = pd.to_datetime("2024-05-01")

# end_used = pd.to_datetime("2022-10-06")
# Calculate the number of days between the dates
days_between = (end_date - start_date).days
n_time_intervals = math.ceil(days_between/ time_interval)
print("number of time intervals = ", n_time_intervals)

# Calculate date ranges
date_ranges = pd.date_range(start=start_date, end=end_date, freq=pd.DateOffset(days=time_interval))
print(len(date_ranges))

# Specify your desired file path
file_path = './Processed_data.pkl'


def log_energy(mag):
    sum_energy = 0
    for m in mag:
        sum_energy += np.power(10, (1.5 * m))
    return np.log10(sum_energy) / 1.5


if not os.path.exists(file_path):
    # Process data for each square
    results = []
    for square in tqdm(list_of_squares, desc="Processing squares"):
        square_data = df[df['square'] == square]
        energies_14 = []
        energies_28 = []
        energies_84 = []
        energies_364 = []
        energies_1820 = []

        counts_14 = []
        counts_28 = []
        counts_84 = []
        counts_364 = []
        counts_1820 = []

        # maxs_14 = []
        # maxs_28 = []
        # maxs_84 = []
        # maxs_364 = []
        # maxs_1820 = []

        for start_day in date_ranges[:-1]:  # Adjusted to avoid accessing beyond the last date
            end_day = start_day + pd.DateOffset(days=time_interval)  # Define the end day for each interval
            days_28 = start_day - pd.DateOffset(days=28 - 14)
            days_84 = start_day - pd.DateOffset(days=84 - 14)
            days_364 = start_day - pd.DateOffset(days=364 - 14)
            days_1820 = start_day - pd.DateOffset(days=1820 - 14)

            interval_data_14 = square_data[(square_data.index >= start_day) & (square_data.index < end_day)]['mag']
            interval_data_28 = square_data[(square_data.index >= days_28) & (square_data.index < end_day)]['mag']
            interval_data_84 = square_data[(square_data.index >= days_84) & (square_data.index < end_day)]['mag']
            interval_data_364 = square_data[(square_data.index >= days_364) & (square_data.index < end_day)]['mag']
            interval_data_1820 = square_data[(square_data.index >= days_1820) & (square_data.index < end_day)]['mag']

            interval_data_14_329 = interval_data_14[interval_data_14 >= 3.29]
            interval_data_28_329 = interval_data_28[interval_data_28 >= 3.29]
            interval_data_84_329 = interval_data_84[interval_data_84 >= 3.29]
            interval_data_364_329 = interval_data_364[interval_data_364 >= 3.29]
            interval_data_1820_329 = interval_data_1820[interval_data_1820 >= 3.29]

            total_energy_14 = log_energy(interval_data_14) if not interval_data_14.empty else 0
            energies_14.append(total_energy_14)
            total_energy_28 = log_energy(interval_data_28) if not interval_data_28.empty else 0
            energies_28.append(total_energy_28)
            total_energy_84 = log_energy(interval_data_84) if not interval_data_84.empty else 0
            energies_84.append(total_energy_84)
            total_energy_364 = log_energy(interval_data_364) if not interval_data_364.empty else 0
            energies_364.append(total_energy_364)
            total_energy_1820 = log_energy(interval_data_1820) if not interval_data_1820.empty else 0
            energies_1820.append(total_energy_1820)

            count_14 = len(interval_data_14_329) if not interval_data_14_329.empty else 0
            counts_14.append(count_14)
            count_28 = len(interval_data_28_329) if not interval_data_28_329.empty else 0
            counts_28.append(count_28)
            count_84 = len(interval_data_84_329) if not interval_data_84_329.empty else 0
            counts_84.append(count_84)
            count_364 = len(interval_data_364_329) if not interval_data_364_329.empty else 0
            counts_364.append(count_364)
            count_1820 = len(interval_data_1820_329) if not interval_data_1820_329.empty else 0
            counts_1820.append(count_1820)

            # max_14 = interval_data_14.max() if not interval_data_14.empty else 0
            # maxs_14.append(max_14)
            # max_28 = interval_data_28.max() if not interval_data_28.empty else 0
            # maxs_28.append(max_28)
            # max_84 = interval_data_84.max() if not interval_data_84.empty else 0
            # maxs_84.append(max_84)
            # max_364 = interval_data_364.max() if not interval_data_364.empty else 0
            # maxs_364.append(max_364)
            # max_1820 = interval_data_1820.max() if not interval_data_1820.empty else 0
            # maxs_1820.append(max_1820)

        square_results = pd.DataFrame({
            'unique_id': ['B' + str(square)] * len(energies_14),
            'ds': date_ranges[1:],
            'start_date': date_ranges[:-1],
            'end_day': date_ranges[1:],

            'y': energies_14,
            'energies_28': energies_28,
            'energies_84': energies_84,
            'energies_364': energies_364,
            'energies_1820': energies_1820,

            'counts_14': counts_14,
            'counts_28': counts_28,
            'counts_84': counts_84,
            'counts_364': counts_364,
            'counts_1820': counts_1820,

            # 'maxs_14': maxs_14,
            # 'maxs_28': maxs_28,
            # 'maxs_84': maxs_84,
            # 'maxs_364': maxs_364,
            # 'maxs_1820': maxs_1820,

        })

        results.append(square_results)

    # Combine all results
    squares_new_samples = pd.concat(results)
    squares_new_samples.reset_index(drop=True, inplace=True)

    # Save the data
    squares_new_samples.to_pickle(file_path)
    print(f'Data saved to {file_path}')
else:
    # load the data
    print('reading data...')
    squares_new_samples = pd.read_pickle(file_path)
    squares_new_samples.reset_index(drop=True, inplace=True)

squares_new_samples




