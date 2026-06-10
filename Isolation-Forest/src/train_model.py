from sklearn.ensemble import IsolationForest
import joblib
import os

def train(X_train):
    model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
    model.fit(X_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/isolation_forest_model.pkl')
    
    return model