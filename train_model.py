import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load cleaned data
df = pd.read_csv("weather_data.csv")

# Ensure columns exist, adjust if necessary
X = df[['humidity', 'wind_mph']]  # Adjust 'wind_speed' to 'wind_mph' or whichever column represents wind speed
y = df['temp_c']  # Adjust 'temperature' to 'temp_c' for temperature in Celsius or the appropriate column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, "weather_model.pkl")
print("Model trained and saved as weather_model.pkl")
