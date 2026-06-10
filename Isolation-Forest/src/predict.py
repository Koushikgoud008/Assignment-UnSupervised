import joblib

def predict_new_data(X_new, scaler):
    model = joblib.load('models/isolation_forest_model.pkl')
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    anomaly_scores = model.decision_function(X_new_scaled)
    return predictions, anomaly_scores