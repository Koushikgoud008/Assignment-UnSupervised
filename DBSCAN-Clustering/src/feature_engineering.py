import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    X = df[['Latitude', 'Longitude']].copy()
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X = X.fillna(X.mean())
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler