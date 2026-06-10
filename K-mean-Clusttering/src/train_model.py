from sklearn.cluster import KMeans
import joblib
import os

def train(X_train):
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(X_train)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/kmeans_model.pkl')
    return model