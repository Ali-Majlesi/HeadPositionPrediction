from HeadPositionPrediction.utils.quaternion import euler_to_quaternion, quaternion_to_euler
import os
from tqdm import tqdm
import pandas as pd
from HeadPositionPrediction.utils.calculate_speed import calculate_speed_acceleration, calculate_angular_velocity

def combine_csv_files(folder_path, relative_to_fisrt_point = False, is_calculate_speed = False, is_calculate_angular_vel=False):
    """
    Combine all CSV files in a specified folder into a single Pandas DataFrame.

    Parameters:
    - folder_path (str): The path to the folder containing the CSV files.

    Returns:
    - combined_df (pd.DataFrame): A DataFrame containing the combined data from all CSV files.
    """
    # Initialize an empty list to store DataFrames from each CSV file
    data_frames = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Loop through each CSV file and read its data into a DataFrame
    for csv_file in tqdm(csv_files):
        file_path = os.path.join(folder_path, csv_file)
        data_frame = pd.read_csv(file_path)
        if relative_to_fisrt_point:
            data_frame['x'] = data_frame['x'] - data_frame['x'].iloc[0]
            data_frame['y'] = data_frame['y'] - data_frame['y'].iloc[0]
            data_frame['z'] = data_frame['z'] - data_frame['z'].iloc[0]

        if is_calculate_speed:
            data_frame = calculate_speed_acceleration(data_frame)

        if is_calculate_angular_vel:
            data_frame = calculate_angular_velocity(data_frame)

        qw = data_frame['qw'].values
        qx = data_frame['qx'].values
        qy = data_frame['qy'].values
        qz = data_frame['qz'].values
        phi, theta, psi = quaternion_to_euler(qx, qy, qz, qw)
        data_frame['phi'] = phi
        data_frame['theta'] = theta
        data_frame['psi'] = psi

        data_frames.append(data_frame.iloc[50:])


    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df