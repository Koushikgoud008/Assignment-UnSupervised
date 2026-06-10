import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    X = df.drop(columns=['Timestamp'], errors='ignore').copy()
    
    age_map = {'Less than 20': 0, '21 to 35': 1, '36 to 50': 2, '51 or more': 3}
    gender_map = {'Female': 0, 'Male': 1}
    
    if 'AGE' in X.columns:
        X['AGE'] = X['AGE'].map(age_map)
    if 'GENDER' in X.columns:
        X['GENDER'] = X['GENDER'].map(gender_map)
        
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X = X.fillna(0)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler