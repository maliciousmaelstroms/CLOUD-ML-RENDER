from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("weather_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "🌤️ Weather Prediction Model is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame([data])  # Convert input to DataFrame
        prediction = model.predict(features)
        return jsonify({"predicted_temp_c": round(prediction[0], 2)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)

