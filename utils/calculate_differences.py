import numpy as np

def calculate_differences(df, dn, columns):
    """
    Calculate the difference between the current row and a row dn rows later for columns x, y, and z in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with columns t, x, y, z.
    - dn (int): The number of rows to shift before calculating the difference.
    - columns (list): A list of column names to calculate difference.

    Returns:
    - df (pd.DataFrame): The DataFrame with added columns for the differences of x, y, and z.
    """
    for column in columns:
        dx = df[column][dn:].values - df[column][:-dn].values
        df['d' + column] = np.append(dx, [0] * dn)

    return df