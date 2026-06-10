import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    X = df[['Instagram visit score', 'Spending_rank(0 to 100)']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler