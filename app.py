from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def index():
    return "Hello from Heroku!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Example: data could have "features" for your model
    # model_prediction = ...
    # Return something
    return jsonify({"prediction": "some_value"})

if __name__ == "__main__":
    app.run(debug=True)
