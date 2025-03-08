from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def index():
    return "Hello from Heroku!"

@app.route("/predict", methods=["GET"])
def predict():
    # 1) Grab the 'song_id' from the URL's query parameters, e.g. /predict?song_id=115
    song_id = request.args.get("song_id")

    # 2) If it's missing or empty, return an error or handle as you wish
    if not song_id:
        return jsonify({"error": "song_id parameter is missing"}), 400

    # 3) Return a JSON with your custom message
    return jsonify({"message": f"Received prediction request for song number {song_id}"})

if __name__ == "__main__":
    # Typically Heroku sets the PORT environment variable, but debug local run uses 5000 by default
    app.run(debug=True)
