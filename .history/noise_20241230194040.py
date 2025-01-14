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
    num_to_flip = int(len(y) * (percentage / 100))
    indices_to_flip = random.sample(range(len(y)), num_to_flip)

    for idx in indices_to_flip:
        y.iloc[idx, 0] = 1 - y.iloc[idx, 0]
    return y