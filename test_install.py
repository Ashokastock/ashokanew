import yfinance
import pandas
import numpy
import tensorflow
import sklearn
import matplotlib
import streamlit
print("All libraries installed successfully!")
import yfinance as yf
import pandas as pd
# Download AAPL data (2010-2025)
data = yf.download("AAPL", start="2010-01-01", end="2025-01-31")
data.to_csv("AAPL.csv")  # Save to CSV

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("AAPL.csv")
prices = df['Close'].values.reshape(-1, 1)

# Normalize data to [0, 1]
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)
# Create sequences (X) and labels (y)
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_prices, window_size=60)

# Split into train/test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv("AAPL.csv")

# Ensure 'Close' is numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])  # Remove invalid rows

# Extract and scale prices
prices = df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)  # No more errors!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))  # Reduces overfitting
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Reverse scaling

# Compare with actual prices
actual_prices = scaler.inverse_transform(y_test)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(actual_prices, color='blue', label='Actual AAPL Price')
plt.plot(predictions, color='red', label='Predicted AAPL Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(actual_prices, predictions)
rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
print(f"MAE: ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")

df['MA20'] = df['Close'].rolling(window=20).mean()
df['RSI'] = ...  # Calculate RSI

# app.py
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained LSTM model
model = load_model('stock_predictor.h5')

st.title("Stock Price Predictor")
ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", "AAPL")

if st.button("Predict"):
    # Fetch data
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    prices = data['Close'].values.reshape(-1, 1)
    
    # Preprocess
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    
    # Predict next day
    last_60_days = scaled_prices[-60:]
    prediction = model.predict(np.array([last_60_days]))
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    
    st.success(f"Predicted next-day price: ${predicted_price:.2f}")

    

