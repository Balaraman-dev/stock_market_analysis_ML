import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Sequence generator
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Configuration
tickers = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL"]
seq_length = 50
data_dir = "../data/company_datasets"
output_dir = "../data"

X_all, y_all = [], []

for ticker in tickers:
    file_path = os.path.join(data_dir, f"{ticker}_historical_data.csv")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)

        if 'Close' not in df.columns:
            print(f"⚠️ 'Close' column missing in {ticker}, skipping.")
            continue

        # Coerce non-numeric values and drop NaNs
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        if len(df) <= seq_length:
            print(f"⚠️ Not enough data for {ticker}, skipping.")
            continue

        close_prices = df['Close'].values.reshape(-1, 1)

        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Sequence creation
        X_ticker, y_ticker = create_sequences(scaled_data, seq_length)

        X_all.append(X_ticker)
        y_all.append(y_ticker)

        print(f"✅ Processed {ticker}: {X_ticker.shape[0]} samples.")

    except Exception as e:
        print(f"❌ Error processing {ticker}: {e}")

# Final concatenation
if X_all and y_all:
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X.npy"), X_all)
    np.save(os.path.join(output_dir, "y.npy"), y_all)

    print(f"\n🎉 Preprocessing complete. Total samples: {X_all.shape[0]}")
    print(f"Saved to: {output_dir}/X.npy and {output_dir}/y.npy")
else:
    print("⚠️ No valid data was processed.")
