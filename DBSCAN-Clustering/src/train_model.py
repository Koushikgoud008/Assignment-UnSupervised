from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

def train(X_train):
    dbscan = DBSCAN(eps=0.25, min_samples=4)
    labels = dbscan.fit_predict(X_train)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, labels)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/dbscan_knn_model.pkl')
    
    return model