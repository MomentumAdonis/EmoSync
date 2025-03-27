# app.py
# This is the main Flask application that provides prediction endpoints.
# It downloads necessary model files from S3, initializes a global predictor,
# and defines API endpoints for changing the prediction method and generating predictions.

import os
from flask import Flask, request, jsonify
from s3utils import download_from_s3        # Utility functions to interact with S3
from predictor import ModelPredictor       # Class that loads and runs the neural network models
from process_prediction import (           # Functions for post-processing prediction results
    map_song_va_to_emotion, 
    prep_playback_sequence, 
    jsonify_playback_sequence,
    smoothen_prediction, 
    amplify_prediction
)
import pandas as pd
import numpy as np

# Create the Flask app instance
app = Flask(__name__)

# Global variable to select the prediction method.
# Allowed values: "tensorflow", "precalculated", "ABC" (if implemented).
prediction_method = "tensorflow"

# 1) Configuration via environment variables.
# These variables allow dynamic configuration of the S3 bucket and model paths.
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "emo-sync-data")
AROUSAL_MODEL_S3 = os.environ.get("AROUSAL_MODEL_S3", "MODEL WEIGHTS/arousal_model.h5")
VALENCE_MODEL_S3 = os.environ.get("VALENCE_MODEL_S3", "MODEL WEIGHTS/valence_model.h5")

# Local paths where the models will be saved after download.
AROUSAL_MODEL_LOCAL = "/tmp/arousal_model.h5"
VALENCE_MODEL_LOCAL = "/tmp/valence_model.h5"

# 2) On startup, download the model files from S3 (only once).
download_from_s3(BUCKET_NAME, AROUSAL_MODEL_S3, AROUSAL_MODEL_LOCAL)
download_from_s3(BUCKET_NAME, VALENCE_MODEL_S3, VALENCE_MODEL_LOCAL)

# 3) Create a global predictor instance using the downloaded model files.
# The ModelPredictor class loads the TensorFlow models from the specified local paths.
predictor = ModelPredictor(AROUSAL_MODEL_LOCAL, VALENCE_MODEL_LOCAL)

# Define a simple root endpoint for testing connectivity.
@app.route("/")
def hello():
    return "Hello from Heroku!"

# Endpoint to change the prediction method.
@app.route("/changeMethod", methods=["GET"])
def change_method():
    """
    Allows the client to change the prediction method.
    Expects a query parameter 'method' (e.g., ?method=precalculated).
    """
    global prediction_method
    new_method = request.args.get("method")
    if not new_method:
        return jsonify({"error": "method parameter is missing"}), 400

    # Optionally add validation for allowed methods here.
    prediction_method = new_method
    return jsonify({"message": f"Prediction method changed to {new_method}"}), 200

# Prediction endpoint.
@app.route("/predict", methods=["GET"])
def predict_endpoint():
    """
    Main prediction endpoint.
    Expects a 'song_id' parameter. Depending on the selected prediction method, it:
      - For "precalculated": downloads and returns a precomputed JSON from S3.
      - For "tensorflow" (default): downloads the song's feature CSV, processes it through the neural network models,
        post-processes the predictions to generate an emotion sequence, and returns the result as JSON.
      - For "ABC": returns a not implemented message.
    """
    song_id = request.args.get("song_id")
    if not song_id:
        return jsonify({"error": "song_id is missing"}), 400

    global prediction_method

    # If the "precalculated" method is selected, fetch the JSON from S3.
    if prediction_method == "precalculated":
        json_s3_key = f"COMPLETED PLAYBACK STRINGS/{song_id}.json"
        local_json_path = f"/tmp/{song_id}.json"
        try:
            download_from_s3(BUCKET_NAME, json_s3_key, local_json_path)
            with open(local_json_path, "r") as f:
                json_str = f.read()
            return json_str, 200
        except Exception as e:
            return jsonify({"error": f"Failed to download precalculated JSON: {str(e)}"}), 500

    # For the "tensorflow" method, run the prediction pipeline.
    elif prediction_method == "tensorflow":
        # 1) Download the features CSV for the song from S3.
        features_s3_key = f"FEATURES PREDICT/{song_id}.csv"
        local_features_path = f"/tmp/{song_id}.csv"
        try:
            download_from_s3(BUCKET_NAME, features_s3_key, local_features_path)
        except Exception as e:
            return jsonify({"error": f"Failed to download features CSV: {str(e)}"}), 500

        # 2) Load the CSV into a DataFrame.
        try:
            df_features = pd.read_csv(local_features_path)
            # Assume the first column is "frameTime" (timestamps) and the rest are feature values.
            times = df_features["frameTime"].astype(float).values
            X = df_features.iloc[:, 1:].values
        except Exception as e:
            return jsonify({"error": f"Error processing features CSV: {str(e)}"}), 500

        # 3) Use the global predictor to generate predictions for arousal and valence.
        aro_pred = predictor.arousal_model.predict(X).flatten()
        val_pred = predictor.valence_model.predict(X).flatten()

        # 4) Combine predictions with timestamps into a DataFrame.
        df_pred = pd.DataFrame({
            "time_s": times,
            "arousal_pred": aro_pred,
            "valence_pred": val_pred
        })

        # 5) Optionally adjust predictions using smoothing and amplification.
        df_pred = smoothen_prediction(df_pred)
        df_pred = amplify_prediction(df_pred)

        # 6) Map predictions to emotion labels and prepare the playback sequence.
        df_emotion = map_song_va_to_emotion(df_pred)
        df_sequence = prep_playback_sequence(df_emotion, minDuration=3.0)

        # 7) Convert the playback sequence to a JSON string.
        json_str = jsonify_playback_sequence(df_sequence, song_id=song_id)
        return json_str, 200

    # If the "ABC" method is selected, return a not-implemented error.
    elif prediction_method == "ABC":
        return jsonify({"error": "ABC prediction method is not yet implemented."}), 501

    else:
        # For any unrecognized prediction method, return an error.
        return jsonify({"error": "Invalid prediction method specified."}), 400

# Entry point for local debugging. When deployed on Heroku, the PORT variable is set differently.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
