import pandas as pd

def load_dataset(filepath):
    return pd.read_csv(filepath)

def save_dataset(df, filepath):
    df.to_csv(filepath, index=False)