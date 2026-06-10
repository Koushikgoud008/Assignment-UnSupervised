import joblib

def predict_new_data(X_new, scaler):
    model = joblib.load('models/tsne_knn_model.pkl')
    X_new_scaled = scaler.transform(X_new)
    components = model.predict(X_new_scaled)
    return components