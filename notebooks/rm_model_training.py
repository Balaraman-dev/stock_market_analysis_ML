import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib  # for saving the model

# Parameters
ticker = "TSLA"
start_date = "2010-01-01"
end_date = "2025-04-01"
seq_length = 50

# Fetch data
stock = yf.Ticker(ticker)
data = stock.history(start=start_date, end=end_date)

if data.empty:
    raise ValueError("No data found for ticker")

close_prices = data['Close'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Prepare sequences
X, y = [], []
for i in range(seq_length, len(scaled_data)):
    X.append(scaled_data[i - seq_length:i].flatten())
    y.append(scaled_data[i][0])
X, y = np.array(X), np.array(y)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model and scaler
joblib.dump(model, "rf_stock_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved!")
