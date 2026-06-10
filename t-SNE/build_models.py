import os
import joblib
from src.data_preprocessing import load_data
from src.feature_engineering import prepare_features
from src.train_model import train

print("Loading raw data...")
df = load_data()

print("Scaling features...")
X_scaled, scaler, feature_cols = prepare_features(df)

print("Training mathematical model...")
train(X_scaled)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')

print("Success! All .pkl files have been generated in the models/ folder.")