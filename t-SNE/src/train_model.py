from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
import joblib
import os

def train(X_train):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_components = tsne.fit_transform(X_train)
    
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train, tsne_components)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/tsne_knn_model.pkl')
    
    return model