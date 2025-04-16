import requests

# Replace with your running Flask URL
url = "http://127.0.0.1:8080/predict"

# Sample input — make sure the keys match your training features
data = {
    "humidity": 75,
    "wind_mph": 10
}

response = requests.post(url, json=data)
print("Server response:", response.json())
