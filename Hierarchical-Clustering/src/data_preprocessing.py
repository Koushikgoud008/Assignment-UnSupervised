import pandas as pd

def load_data(filepath='./data/dataset.csv'):
    df = pd.read_csv(filepath)
    return df