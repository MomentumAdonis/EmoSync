import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

class ModelPredictor:
    """
    A simple class that loads the trained models for arousal/valence
    and provides a method to run predictions on a batch of feature vectors.
    """
    def __init__(self, arousal_model_path, valence_model_path):
        """
        Loads the trained models from the given paths and compiles them.
        """
        self.arousal_model = load_model(arousal_model_path, compile=False)
        self.valence_model = load_model(valence_model_path, compile=False)

        # Compile the models so that .evaluate() works (not strictly needed for .predict())
        self.arousal_model.compile(optimizer=Adam(learning_rate=0.001),
                                   loss='mse',
                                   metrics=['mean_absolute_error'])
        self.valence_model.compile(optimizer=Adam(learning_rate=0.001),
                                   loss='mse',
                                   metrics=['mae'])

    def predict_whole_song_X_all(self, X_all, times_all):
        """
        For a given feature array X (for a single song) and corresponding times, predict arousal and valence
        using the loaded neural network models, and return a DataFrame with columns:
        time_s, arousal_pred, valence_pred.

        Parameters:
        -----------
        X : np.ndarray
            2D array of features (each row corresponds to a time step).
        times : np.ndarray
            1D array of time stamps (in seconds) corresponding to each row in X.
    
        Returns:
        --------
        df_predictions : pd.DataFrame
            DataFrame with columns "time_s", "arousal_pred", and "valence_pred", sorted by time.
        """

        # Use the instance models to predict
        aro_pred = self.arousal_model.predict(X_all).flatten()
        val_pred = self.valence_model.predict(X_all).flatten()

        # Sort by time
        sort_idx = np.argsort(times_all)
        times_sorted = times_all[sort_idx]
        aro_pred_sorted = aro_pred[sort_idx]
        val_pred_sorted = val_pred[sort_idx]

        # Create a DataFrame with the results
        df_predictions = pd.DataFrame({
            "time_s": times_sorted,
            "arousal_pred": aro_pred_sorted,
            "valence_pred": val_pred_sorted
        })

        return df_predictions

