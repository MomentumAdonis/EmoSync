Readme will be completed later
# EmoSync Backend Server

This repository contains the backend server code for the EmoSync project. The server is built with Flask and is designed to be deployed on Heroku using Gunicorn as the WSGI server. It supports predicting emotion sequences for songs using pre-trained neural network models, processing these predictions into emotion sequences, and serving the results as JSON.

## Files Overview

- **app.py**  
  The main Flask application that defines API endpoints. It handles:
  - Changing the prediction method via `/changeMethod`
  - Predicting emotion sequences for a given song via `/predict`
  - Downloading required model and feature files from S3 on startup

- **predictor.py**  
  Contains the `ModelPredictor` class which loads and compiles the pre-trained models for arousal and valence, and provides a method to predict emotion values for a whole song.

- **process_prediction.py**  
  Implements functions to:
  - Map valence–arousal predictions to discrete emotion labels (using a circular segmentation approach)
  - Smooth and amplify prediction data
  - Prepare and compress playback sequences
  - Convert playback sequences into JSON format

- **s3utils.py**  
  Provides helper functions to interact with Amazon S3:
  - Create an S3 client using environment variables
  - Download files from S3
  - Upload files to S3
  - List files in an S3 bucket

- **Procfile**  
  Contains the command for running the server with Gunicorn:  
  `web: gunicorn app:app`

- **requirements.txt**  
  Lists the required Python packages:



## Deployment

The application is configured to be deployed on Heroku. The Procfile specifies that Gunicorn should run the Flask app defined in `app.py`. Environment variables (e.g., AWS credentials, S3 bucket names, and model paths) are managed via Heroku config or a `.env` file.

## Usage

1. **API Endpoints:**
 - **/**: Returns a simple greeting message.
 - **/changeMethod**: Accepts a `method` query parameter to change the prediction method.
 - **/predict**: Accepts a `song_id` query parameter and returns a JSON string containing the predicted emotion sequence for the song. Depending on the prediction method, it either runs the prediction pipeline using TensorFlow or fetches precalculated JSON data from S3.

2. **Local Testing:**
 - Run the app locally with `python app.py`.
 - Access the endpoints via your browser or a tool like cURL/Postman.

## Summary

This backend server integrates pre-trained neural network models for emotion recognition from music with S3 for storage and retrieval of model and feature files. It processes predictions to generate emotion sequences and serves them through a simple Flask API. The design is modular and ready for deployment on Heroku.
