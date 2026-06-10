from sklearn.mixture import GaussianMixture
import joblib
import os

def train(X_train):
    model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    model.fit(X_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/gmm_model.pkl')
    
    return model