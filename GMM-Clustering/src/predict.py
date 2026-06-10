import joblib

def predict_new_data(X_new, scaler):
    model = joblib.load('models/gmm_model.pkl')
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    probabilities = model.predict_proba(X_new_scaled)
    return predictions, probabilities