from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib
import os

def train(X_train):
    hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
    labels = hc.fit_predict(X_train)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, labels)
    
    centers = np.array([X_train[labels == i].mean(axis=0) for i in range(3)])
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/hc_full_model.pkl')
    joblib.dump(centers, 'models/hc_full_centers.pkl')
    
    return model, centers