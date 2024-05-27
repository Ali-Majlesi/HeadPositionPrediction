import pandas as pd
import os
from HeadPositionPrediction.utils.quaternion import euler_to_quaternion
import numpy as np

def adjust_sample_rate(data_path, output_path, interpolation_time):
    # Create output directory if not exist
    os.makedirs(output_path, exist_ok=True)

    # Create Train, Test and Valid subdirectories
    sub_dirs = ['Train', 'Test', 'Valid']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)

    for subfolder in sub_dirs:
        folder_path = os.path.join(data_path, subfolder)
        # List all CSV files in the data directory
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            save_folder = os.path.join(output_path, subfolder)

            file_path_src = os.path.join(folder_path, csv_file)
            file_path_dst = os.path.join(save_folder, csv_file)
            data_frame = pd.read_csv(file_path_src)

            q = euler_to_quaternion(data_frame['yaw'].values*np.pi/180, data_frame['pitch'].values*np.pi/180, data_frame['roll'].values*np.pi/180)

            # Remove the specified columns
            data_frame = data_frame.drop(columns=['pitch', 'yaw', 'roll'])

            # Add new columns qx, qy, qz, and qw
            data_frame = data_frame.assign(qx=q[:, 0], qy=q[:, 1], qz=q[:, 2], qw=q[:, 3])

            # Convert 't' column to datetime if it's not already
            data_frame['t'] = pd.to_datetime(data_frame['t'], unit='s')

            # Set 't' as the index
            data_frame.set_index('t', inplace=True)

            # Resample to 0.01 seconds
            resampled_df = data_frame.resample(interpolation_time).mean() # '10L' stands for 10 milliseconds, which is 0.01 seconds

            # Interpolate missing values
            interpolated_df = resampled_df.interpolate(method='time')

            # Convert the DatetimeIndex back to floating point numbers (seconds)
            # Subtract a base datetime to get Timedelta objects, then apply total_seconds()
            interpolated_df.index = (interpolated_df.index - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')

            # Reset the index to make 't' a column again
            interpolated_df.reset_index(inplace=True)

            # Rename the index column to 't'
            interpolated_df.rename(columns={'index': 't'}, inplace=True)

            # Save to CSV
            interpolated_df.to_csv(file_path_dst, index=False)