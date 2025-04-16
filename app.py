from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("weather_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")  # Renders the frontend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()
        humidity = data['humidity']
        wind_speed = data['wind_speed']  # Sent from frontend

        # Match the model’s training column names exactly
        features = pd.DataFrame([[humidity, wind_speed]], columns=['humidity', 'wind_mph'])

        # Make prediction
        prediction = model.predict(features)
        predicted_temp = round(prediction[0], 2)

        return jsonify({'predicted_temp_c': predicted_temp})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)



