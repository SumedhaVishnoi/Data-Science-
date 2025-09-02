import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Load dataset (example: Apple stock)
df = pd.read_csv("AAPL.csv")  # Assume file has 'Date' and 'Close'
data = df['Close'].values.reshape(-1,1)

# Normalize data (0 to 1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create sequences
X, y = [], []
time_steps = 60   # last 60 days used to predict next day
for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape for LSTM [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))  # Predict next closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

# Predict
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(data, color='blue', label='Real Stock Price')
plt.plot(np.arange(time_steps, len(predicted)+time_steps), predicted, color='red', label='Predicted Price')
plt.legend()
plt.show()
