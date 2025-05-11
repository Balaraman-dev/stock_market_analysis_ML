import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load raw data
data = pd.read_csv("../data/historical_data.csv")

# Select 'Close' column for prediction
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize data using Min-Max Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create sequences (sliding window approach)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 50  # Use past 60 days to predict the next day
X, y = create_sequences(scaled_data, seq_length)

# Save preprocessed data
np.save("../data/X.npy", X)
np.save("../data/y.npy", y)
print("Preprocessing complete. Files saved as data/X.npy and data/y.npy.")