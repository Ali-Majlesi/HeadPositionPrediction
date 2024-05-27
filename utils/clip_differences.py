def clip_differences(df, columns, lower_bound_percentile=0.025, upper_bound_percentile=0.975, lower_bounds=None, upper_bounds=None):
    """
    Clip the differences in the specified columns to be within the lower bound of the 2.5 percentile and the higher bound of the 97.5 percentile.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the differences.
    - columns (list): A list of column names to clip.
    - lower_bound_percentile (float): The lower bound percentile (default is 2.5%).
    - upper_bound_percentile (float): The upper bound percentile (default is 97.5%).

    Returns:
    - df (pd.DataFrame): The DataFrame with clipped differences.
    """
    if lower_bounds is None:
        lower_bounds = {}
        for column in columns:
            lower_quantile = df[column].quantile(lower_bound_percentile)
            lower_bounds[column] = lower_quantile

    if upper_bounds is None:
        upper_bounds = {}
        for column in columns:
            upper_quantile = df[column].quantile(upper_bound_percentile)
            upper_bounds[column] = upper_quantile

    for column in columns:
        df[column] = df[column].clip(lower=lower_bounds[column], upper=upper_bounds[column])

    return df, (lower_bounds, upper_bounds)