import joblib

def predict_new_data(X_new, scaler):
    pca = joblib.load('models/pca_model.pkl')
    X_new_scaled = scaler.transform(X_new)
    components = pca.transform(X_new_scaled)
    explained_variance = pca.explained_variance_ratio_
    return components, explained_variance