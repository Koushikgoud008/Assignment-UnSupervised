import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    col_map = {}
    for expected, fallbacks in {
        'gearbox_c': ['gearbox_c', 'gearbox_c_temp', 'gearbox_casing'],
        'gearbox_b': ['gearbox_b', 'gearbox_b_temp', 'gearbox_bearing'],
        'vibration_x': ['vibration_x'],
        'vibration_y': ['vibration_y'],
        'vibration_z': ['vibration_z'],
        'oil_pressure': ['oil_pressure', 'oil_pressur'],
        'particle_count': ['particle_count', 'particle_c']
    }.items():
        for f in fallbacks:
            if f in df.columns:
                col_map[f] = expected
                break

    X = df[list(col_map.keys())].copy()
    X.rename(columns=col_map, inplace=True)
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X = X.fillna(X.median())
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, X.columns