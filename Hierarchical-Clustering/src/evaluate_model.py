from sklearn.metrics import silhouette_score

def evaluate(model, X_train):
    labels = model.labels_
    inertia = model.inertia_
    sil_score = silhouette_score(X_train, labels)
    return inertia, sil_score