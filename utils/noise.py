import random
import pandas as pd

def flip_labels(y, percentage):
    """
    Flips a specific percentage of labels in the given dataset.

    Parameters:
    y (pd.DataFrame): The DataFrame containing the labels.
    percentage (float): The percentage of labels to flip (between 0 and 100).

    Returns:
    pd.DataFrame: The DataFrame with flipped labels.
    """
    n = len(y)
    m = int(n * percentage / 100)
    indices = random.sample(range(n), m)
    y_flipped = y.copy()
    y_flipped.iloc[indices] = -y_flipped.iloc[indices]
    return y_flipped