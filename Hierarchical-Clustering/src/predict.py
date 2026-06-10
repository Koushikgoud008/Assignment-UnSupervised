import joblib

def predict_new_data(X_new, scaler):
    model = joblib.load('models/hc_model.pkl')
    X_new_scaled = scaler.transform(X_new)
    predictions = model.predict(X_new_scaled)
    return predictions