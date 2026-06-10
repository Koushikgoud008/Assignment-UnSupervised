from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import joblib
import os

def train(X_train):
    # Industry-standard optimization for heavy t-SNE computation
    MAX_SAMPLES = 10000
    
    if X_train.shape[0] > MAX_SAMPLES:
        print(f"Dataset too large for standard t-SNE ({X_train.shape[0]} rows). Sub-sampling to {MAX_SAMPLES} rows for performance...")
        np.random.seed(42)
        indices = np.random.choice(X_train.shape[0], MAX_SAMPLES, replace=False)
        X_train_sampled = X_train[indices, :]
    else:
        X_train_sampled = X_train

    print("Running t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_components = tsne.fit_transform(X_train_sampled)
    
    print("Training KNN Regressor on embedded manifold...")
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train_sampled, tsne_components)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/tsne_knn_model.pkl')
    
    return model