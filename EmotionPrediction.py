import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import sys
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import boto3


features_long_filtered_path = 'TRAINING DATA/features_long_filtered.csv'
valence_long_filtered_path = 'TRAINING DATA/valence_long_filtered.csv'
arousal_long_filtered_path = 'TRAINING DATA/arousal_long_filtered.csv'


s3 = boto3.client(
    's3',
    aws_access_key_id='AKIASK5MCSKPTZUKOIV2',
    aws_secret_access_key='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',
    region_name='eu-north-1'
)

def download_from_s3(bucket_name, s3_key, local_path):
    """Download a file from S3 to local_path."""
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")

# Example usage:
BUCKET_NAME = "emo-sync-data"
download_from_s3(BUCKET_NAME, arousal_long_filtered_path, "/tmp/arousal_long_filtered.csv")
download_from_s3(BUCKET_NAME, valence_long_filtered_path, "/tmp/valence_long_filtered.csv")
download_from_s3(BUCKET_NAME, features_long_filtered_path, "/tmp/features_long_filtered.csv")


#Reads the dataset and loads it into a pandas frame
features_long_filtered = pd.read_csv("/tmp/features_long_filtered.csv")
valence_long_filtered = pd.read_csv("/tmp/valence_long_filtered.csv")
arousal_long_filtered = pd.read_csv("/tmp/arousal_long_filtered.csv")

print(features_long_filtered.shape)
print(valence_long_filtered.shape)
print(arousal_long_filtered.shape)

print(features_long_filtered.head(10))
print(valence_long_filtered.head(10))
print(arousal_long_filtered.head(10))


# ----- MERGE DATAFRAMES -----
# Merge features_long_filtered and arousal_long_filtered on ['song_id', 'time_s']
data_temp = pd.merge(features_long_filtered, arousal_long_filtered[['song_id', 'time_s', 'arousal_value']], 
                     on=['song_id', 'time_s'], how='inner')

# Merge the result with valence_long_filtered on ['song_id', 'time_s']
data = pd.merge(data_temp, valence_long_filtered[['song_id', 'time_s', 'valence_value']],
                on=['song_id', 'time_s'], how='inner')

#Extract the metadata: (song_id, time_s)
meta = data[['song_id', 'time_s']].values

# Check the merged DataFrame shape and head (for debugging)
print("Merged data shape:", data.shape)
print(data.head())


# ----- EXTRACT FEATURES (X) AND TARGETS (y) -----
# Assume columns: 0:"song_id", 1:"time_s", 2:"frameTime", 3:... are the 260 feature columns.
# If that's the case, then:
X = data.iloc[:, 3:3+260].values  # adjust if there are extra columns
y_arousal = data['arousal_value'].values
y_valence = data['valence_value'].values

print("X shape:", X.shape)
print("y_arousal shape:", y_arousal.shape)
print("y_valence shape:", y_valence.shape)

# Optionally, standardize X (for example, using mean=0 and std=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# ----- SPLIT DATA INTO TRAIN AND TEST SETS -----
X_train, X_test, y_arousal_train, y_arousal_test, meta_train, meta_test = train_test_split(
    X_scaled, y_arousal, meta, test_size=0.2, random_state=42
)
_, _, y_valence_train, y_valence_test, _, _ = train_test_split(
    X_scaled, y_valence, meta, test_size=0.2, random_state=42
)

print("Shapes:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_arousal_train:", y_arousal_train.shape, "y_arousal_test:", y_arousal_test.shape)
print("meta_train:", meta_train.shape, "meta_test:", meta_test.shape)



# ----- DEFINE A SIMPLE NEURAL NETWORK MODEL -----
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))  # regression output
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Build model for arousal prediction
arousal_model = build_model(input_dim=X_scaled.shape[1])
print(arousal_model.summary())

# Build model for valence prediction
valence_model = build_model(input_dim=X_scaled.shape[1])
print(valence_model.summary())



# ----- TRAIN THE MODELS -----
# For arousal
arousal_history = arousal_model.fit(X_train, y_arousal_train, epochs=50, batch_size=32,
                                    validation_data=(X_test, y_arousal_test))

# For valence
valence_history = valence_model.fit(X_train, y_valence_train, epochs=50, batch_size=32,
                                    validation_data=(X_test, y_valence_test))



# ----- EVALUATE MODELS -----
arousal_loss, arousal_mae = arousal_model.evaluate(X_test, y_arousal_test)
valence_loss, valence_mae = valence_model.evaluate(X_test, y_valence_test)
print("Arousal - Test Loss (MSE):", arousal_loss, "MAE:", arousal_mae)
print("Valence - Test Loss (MSE):", valence_loss, "MAE:", valence_mae)



def test_random_samples(num_samples=3):
    """
    Pick random rows from X_test and compare predicted vs. actual 
    for both arousal and valence, including song_id and time_s.
    """
    for _ in range(num_samples):
        # Random index from test set
        idx = random.randrange(len(X_test))
        
        # Extract the features for this single test sample
        sample_features = X_test[idx].reshape(1, -1)
        
        # Predict arousal & valence
        arousal_pred = arousal_model.predict(sample_features)[0][0]
        valence_pred = valence_model.predict(sample_features)[0][0]
        
        # Actual ground truth
        arousal_true = y_arousal_test[idx]
        valence_true = y_valence_test[idx]
        
        # Song ID and time from metadata
        song_id = meta_test[idx][0]
        time_s = meta_test[idx][1]
        
        print(f"--- Random Test Sample (index {idx}) ---")
        print(f"Song ID: {song_id}, Time: {time_s:.1f}s")
        print(f"Arousal -> Predicted: {arousal_pred:.4f}, Actual: {arousal_true:.4f}")
        print(f"Valence -> Predicted: {valence_pred:.4f}, Actual: {valence_true:.4f}")
        print()


# Example usage
test_random_samples(num_samples=3)


def test_whole_song(song_id):
    """
    For a given song_id, gather all test rows from X_test, meta_test,
    y_arousal_test, y_valence_test. Predict arousal/valence, then plot
    predicted vs. actual lines over time.
    """

    # 1. Find indices in meta_test that match this song_id
    idx_list = np.where(meta_test[:, 0] == song_id)[0]
    if len(idx_list) == 0:
        print(f"No test samples found for song_id={song_id} in the test set.")
        return
    
    # 2. Gather the relevant rows
    X_song = X_test[idx_list]
    times = meta_test[idx_list, 1]
    aro_true = y_arousal_test[idx_list]
    val_true = y_valence_test[idx_list]
    
    # 3. Predict arousal & valence
    aro_pred = arousal_model.predict(X_song).flatten()
    val_pred = valence_model.predict(X_song).flatten()
    
    # 4. Sort by time so the line chart is in ascending time order
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    aro_true = aro_true[sort_idx]
    val_true = val_true[sort_idx]
    aro_pred = aro_pred[sort_idx]
    val_pred = val_pred[sort_idx]
    
    # 5. Plot
    plt.figure(figsize=(8,6))
    plt.title(f"Song {song_id}")
    
    # Set y-limits for convenience, e.g. [-1.0, 1.0]
    plt.ylim(-1.0, 1.0)
    
    # Plot predicted vs. actual Arousal
    plt.plot(times, aro_pred, color='red', alpha=0.5, label='Pred Arousal')
    plt.plot(times, aro_true, color='red', label='True Arousal')
    
    # Plot predicted vs. actual Valence
    plt.plot(times, val_pred, color='blue', alpha=0.5, label='Pred Valence')
    plt.plot(times, val_true, color='blue', label='True Valence')
    
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


test_whole_song(2057)