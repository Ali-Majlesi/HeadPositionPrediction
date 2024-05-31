import numpy as np

def windowing(df, column_names, window_size, pred_horizen=1, shift_output=True):
    """
    Create sequences of data points from specified columns, window them, and return input sequences (X) and target values (Y).

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the time series data.
    - column_names (list): A list of column names to be used for creating the sequences.
    - window_size (int): The size of the window for creating sequences.

    Returns:
    - X (np.array): The input sequences for the RNN.
    - Y (np.array): The target values for the RNN.
    """
    X = np.zeros((len(df), window_size, len(column_names)))
    Y = np.zeros((len(df), window_size, len(column_names)))

    if shift_output:
        offset_output = window_size
    else:
        offset_output = pred_horizen

    for n_shift in range(window_size):
        X[:, n_shift,:] = df[column_names].shift(-n_shift)
        Y[:, n_shift,:] = df[column_names].shift(-n_shift-offset_output)

    X = X[1:-window_size-offset_output, :,:]
    Y = Y[1:-window_size-offset_output, :,:]

    assert not(np.isnan(Y).any())

    return X, Y