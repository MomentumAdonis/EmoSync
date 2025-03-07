# LoadAndPredict.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt

import boto3
import os

from tensorflow.keras.models import load_model

# -------------------------------------------------------------------------
# 1) Configure S3 client with your credentials
# -------------------------------------------------------------------------
s3 = boto3.client(
    's3',
    aws_access_key_id='AKIASK5MCSKPTZUKOIV2',
    aws_secret_access_key='KOuRUKUy9Lcxdm5MUAE/aFINMZtdC/M7C2C3g633',
    region_name='eu-north-1'
)

def download_from_s3(bucket_name, s3_key, local_path):
    """Download a file from S3 to local_path."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)  # Ensure local folder exists
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")

# -------------------------------------------------------------------------
# 2) Download CSVs from S3 and read into DataFrames
# -------------------------------------------------------------------------
BUCKET_NAME = "emo-sync-data"

arousal_csv_s3 = "TRAINING DATA/arousal_long_filtered.csv"
valence_csv_s3 = "TRAINING DATA/valence_long_filtered.csv"
features_csv_s3 = "TRAINING DATA/features_long_filtered.csv"

download_from_s3(BUCKET_NAME, arousal_csv_s3, "/tmp/arousal_long_filtered.csv")
download_from_s3(BUCKET_NAME, valence_csv_s3, "/tmp/valence_long_filtered.csv")
download_from_s3(BUCKET_NAME, features_csv_s3, "/tmp/features_long_filtered.csv")

arousal_long_filtered = pd.read_csv("/tmp/arousal_long_filtered.csv")
valence_long_filtered = pd.read_csv("/tmp/valence_long_filtered.csv")
features_long_filtered = pd.read_csv("/tmp/features_long_filtered.csv")

print("Shapes of CSVs:")
print("arousal:", arousal_long_filtered.shape)
print("valence:", valence_long_filtered.shape)
print("features:", features_long_filtered.shape)

# -------------------------------------------------------------------------
# 3) Merge into one DataFrame
# -------------------------------------------------------------------------
data_temp = pd.merge(
    features_long_filtered,
    arousal_long_filtered[['song_id', 'time_s', 'arousal_value']],
    on=['song_id', 'time_s'], how='inner'
)
data = pd.merge(
    data_temp,
    valence_long_filtered[['song_id', 'time_s', 'valence_value']],
    on=['song_id','time_s'], how='inner'
)

# Extract metadata
meta = data[['song_id', 'time_s']].values

# The first 3 columns are song_id, time_s, frameTime, so features start at column index 3
X = data.iloc[:, 3:3+260].values
y_arousal = data['arousal_value'].values
y_valence = data['valence_value'].values

print("Merged data shape:", data.shape)
print("X shape:", X.shape)
print("y_arousal shape:", y_arousal.shape)
print("y_valence shape:", y_valence.shape)

# -------------------------------------------------------------------------
# 4) Standardize features
# -------------------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# For demonstration, we do a train_test_split for a test set to do random-sample or whole-song checks
X_train, X_test, y_arousal_train, y_arousal_test, meta_train, meta_test = train_test_split(
    X_scaled, y_arousal, meta, test_size=0.2, random_state=42
)
_, _, y_valence_train, y_valence_test, _, _ = train_test_split(
    X_scaled, y_valence, meta, test_size=0.2, random_state=42
)

print("Test shapes:")
print("X_test:", X_test.shape)
print("y_arousal_test:", y_arousal_test.shape)
print("meta_test:", meta_test.shape)

# -------------------------------------------------------------------------
# 5) Download model weights from S3 and load them
# -------------------------------------------------------------------------
download_from_s3(BUCKET_NAME, "MODEL WEIGHTS/arousal_model.h5", "/tmp/arousal_model.h5")
download_from_s3(BUCKET_NAME, "MODEL WEIGHTS/valence_model.h5", "/tmp/valence_model.h5")

arousal_model = load_model("/tmp/arousal_model.h5", compile=False)
# Now manually compile with a recognized metric
arousal_model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

valence_model = load_model("/tmp/valence_model.h5", compile=False)
# Now manually compile with a recognized metric
valence_model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

print("Models loaded successfully.")

# -------------------------------------------------------------------------
# 6) Prediction Functions
# -------------------------------------------------------------------------
def test_random_samples(num_samples=3):
    """
    Pick random rows from X_test and compare predicted vs. actual 
    for both arousal and valence, including song_id and time_s.
    """
    for _ in range(num_samples):
        idx = random.randrange(len(X_test))
        sample_features = X_test[idx].reshape(1, -1)
        
        aro_pred = arousal_model.predict(sample_features)[0][0]
        val_pred = valence_model.predict(sample_features)[0][0]
        
        aro_true = y_arousal_test[idx]
        val_true = y_valence_test[idx]
        
        song_id = meta_test[idx][0]
        time_s = meta_test[idx][1]
        
        print(f"--- Random Test Sample (index {idx}) ---")
        print(f"Song ID: {song_id}, Time: {time_s:.1f}s")
        print(f"Arousal -> Predicted: {aro_pred:.4f}, Actual: {aro_true:.4f}")
        print(f"Valence -> Predicted: {val_pred:.4f}, Actual: {val_true:.4f}")
        print()

def test_whole_song(song_id):
    """
    For a given song_id, gather all test rows from X_test, meta_test,
    y_arousal_test, y_valence_test. Predict arousal/valence, then plot
    predicted vs. actual lines over time.
    """
    idx_list = np.where(meta_test[:, 0] == song_id)[0]
    if len(idx_list) == 0:
        print(f"No test samples found for song_id={song_id} in the test set.")
        return
    
    X_song = X_test[idx_list]
    times = meta_test[idx_list, 1]
    aro_true = y_arousal_test[idx_list]
    val_true = y_valence_test[idx_list]
    
    aro_pred = arousal_model.predict(X_song).flatten()
    val_pred = valence_model.predict(X_song).flatten()
    
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    aro_true = aro_true[sort_idx]
    val_true = val_true[sort_idx]
    aro_pred = aro_pred[sort_idx]
    val_pred = val_pred[sort_idx]
    
    plt.figure(figsize=(8,6))
    plt.title(f"Song {song_id}")
    plt.ylim(-1.0, 1.0)
    
    plt.plot(times, aro_pred, color='red', alpha=0.5, label='Pred Arousal')
    plt.plot(times, aro_true, color='red', label='True Arousal')
    plt.plot(times, val_pred, color='blue', alpha=0.5, label='Pred Valence')
    plt.plot(times, val_true, color='blue', label='True Valence')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# Example usage (uncomment if you want to run them automatically):
test_random_samples(num_samples=3)
test_whole_song(2001)
