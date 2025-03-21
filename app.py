# app.py
import os
from flask import Flask, request, jsonify
from s3utils import download_from_s3
from predictor import ModelPredictor
from process_prediction import (
    map_song_va_to_emotion, prep_playback_sequence, jsonify_playback_sequence,
    smoothen_prediction, amplify_prediction
)
import pandas as pd
import numpy as np

app = Flask(__name__)

# Global variable to select the prediction method.
# Allowed values: "tensorflow", "precalculated", "ABC" (if implemented)
prediction_method = "precalcualted"

# 1) Configuration via environment variables
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "emo-sync-data")
AROUSAL_MODEL_S3 = os.environ.get("AROUSAL_MODEL_S3", "MODEL WEIGHTS/arousal_model.h5")
VALENCE_MODEL_S3 = os.environ.get("VALENCE_MODEL_S3", "MODEL WEIGHTS/valence_model.h5")

AROUSAL_MODEL_LOCAL = "/tmp/arousal_model.h5"
VALENCE_MODEL_LOCAL = "/tmp/valence_model.h5"

# 2) On startup, download the model files from S3 (only once)
download_from_s3(BUCKET_NAME, AROUSAL_MODEL_S3, AROUSAL_MODEL_LOCAL)
download_from_s3(BUCKET_NAME, VALENCE_MODEL_S3, VALENCE_MODEL_LOCAL)

# 3) Create a global predictor instance (for the TensorFlow method)
predictor = ModelPredictor(AROUSAL_MODEL_LOCAL, VALENCE_MODEL_LOCAL)

@app.route("/")
def hello():
    return "Hello from Heroku!"

@app.route("/changeMethod", methods=["GET"])
def change_method():
    """
    Endpoint to change the prediction method.
    Expects a query parameter 'method', e.g., ?method=precalculated
    """
    global prediction_method
    new_method = request.args.get("method")
    if not new_method:
        return jsonify({"error": "method parameter is missing"}), 400

    # You can add validation here if needed
    prediction_method = new_method
    return jsonify({"message": f"Prediction method changed to {new_method}"}), 200

@app.route("/predict", methods=["GET"])
def predict_endpoint():
    """
    Prediction endpoint that expects a 'song_id' parameter.
    Depending on the global prediction_method, it either:
      - (tensorflow) downloads the song's features CSV, runs predictions using the neural networks,
        processes the prediction into an emotion sequence, and returns a JSON string.
      - (precalculated) downloads a precalculated JSON file from S3 (from COMPLETED PLAYBACK STRINGS) and returns it.
      - (ABC) if implemented, would run that method.
    """
    song_id = request.args.get("song_id")
    if not song_id:
        return jsonify({"error": "song_id is missing"}), 400

    global prediction_method

    # If using the "precalculated" method, simply fetch the corresponding JSON file from S3.
    if prediction_method == "tensorflow":
        json_s3_key = f"COMPLETED PLAYBACK STRINGS/{song_id}.json"
        local_json_path = f"/tmp/{song_id}.json"
        try:
            download_from_s3(BUCKET_NAME, json_s3_key, local_json_path)
            with open(local_json_path, "r") as f:
                json_str = f.read()
            return json_str, 200
        except Exception as e:
            return jsonify({"error": f"Failed to download precalculated JSON: {str(e)}"}), 500

    # Else if using the "tensorflow" method (default), run the prediction pipeline.
    elif prediction_method == "tensorflow":
        # Download the features CSV for the selected song from S3
        features_s3_key = f"FEATURES PREDICT/{song_id}.csv"
        local_features_path = f"/tmp/{song_id}.csv"
        try:
            download_from_s3(BUCKET_NAME, features_s3_key, local_features_path)
        except Exception as e:
            return jsonify({"error": f"Failed to download features CSV: {str(e)}"}), 500

        # Load the features into a DataFrame
        df_features = pd.read_csv(local_features_path)
        # Assume df_features has columns: [frameTime, f0, f1, ..., fN]
        # Use the "frameTime" column for timestamps and all other columns as features.
        try:
            times = df_features["frameTime"].astype(float).values
            X = df_features.iloc[:, 1:].values  # all columns except "frameTime"
        except Exception as e:
            return jsonify({"error": f"Error processing features CSV: {str(e)}"}), 500

        # Run predictions using the global predictor
        aro_pred = predictor.arousal_model.predict(X).flatten()
        val_pred = predictor.valence_model.predict(X).flatten()

        # Combine predictions into a DataFrame
        df_pred = pd.DataFrame({
            "time_s": times,
            "arousal_pred": aro_pred,
            "valence_pred": val_pred
        })

        # Optionally adjust predictions (e.g., smoothing, amplification)
        df_pred = smoothen_prediction(df_pred)
        df_pred = amplify_prediction(df_pred)

        # Convert predictions to an emotion sequence
        df_emotion = map_song_va_to_emotion(df_pred)
        df_sequence = prep_playback_sequence(df_emotion, minDuration=3.0)

        # Convert the sequence to JSON
        json_str = jsonify_playback_sequence(df_sequence, song_id=song_id)
        return json_str, 200

    # If using the "ABC" method (if implemented), you can add that branch here.
    elif prediction_method == "ABC":
        return jsonify({"error": "ABC prediction method is not yet implemented."}), 501

    else:
        return jsonify({"error": "Invalid prediction method specified."}), 400

if __name__ == "__main__":
    # For local debugging (Heroku sets PORT differently)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
