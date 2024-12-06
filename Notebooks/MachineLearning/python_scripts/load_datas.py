
import pandas as pd

def load_data_train():
    return (pd.read_csv('data/X_train.csv'),pd.read_csv('data/y_train.csv')['grav'])


def load_data_test():
    return (pd.read_csv('data/X_test.csv'),pd.read_csv('data/y_test.csv')['grav'])


def load_data_resampled():
    return (pd.read_csv('data/X_train_resampled.csv'),pd.read_csv('data/y_train_resampled.csv')['grav'])