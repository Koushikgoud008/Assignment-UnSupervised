import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    X = df.drop(columns=['CUST_ID'], errors='ignore').copy()
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X = X.fillna(X.median())
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, X.columns