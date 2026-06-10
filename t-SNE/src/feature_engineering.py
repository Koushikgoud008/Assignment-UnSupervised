import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    col_map = {}
    for expected, fallbacks in {
        'mehsul_qiymeti': ['mehsul_qi', 'mehsul_qiymeti'],
        'magaza_lat': ['magaza_la', 'magaza_lat'],
        'magaza_lon': ['magaza_lo', 'magaza_lon'],
        'bonus_kart': ['bonus_kar', 'bonus_kart']
    }.items():
        for f in fallbacks:
            if f in df.columns:
                col_map[f] = expected
                break

    X = df[list(col_map.keys())].copy()
    X.rename(columns=col_map, inplace=True)
    
    X['bonus_kart'] = X['bonus_kart'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0, 1: 1, 0: 0}).fillna(0)
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    X = X.fillna(X.median())
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler, X.columns