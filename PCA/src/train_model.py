from sklearn.decomposition import PCA
import joblib
import os

def train(X_train):
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(pca, 'models/pca_model.pkl')
    
    return pca