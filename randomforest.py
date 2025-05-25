# Import necessary libraries
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Streamlit app
st.title("Stock Market Prediction App (Random Forest)")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA):")

start_date = "2010-01-01"
end_date = "2025-04-01"

if ticker:
    ticker = ticker.strip().upper() 

    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            st.error("No data found for the given stock ticker. Please check the ticker symbol.")
            st.stop()

        st.success(f"Data fetched successfully for {ticker}")
        st.write("Sample Data:", data.tail())  # ✅ Debug info

        close_prices = data['Close'].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Prepare training data
        seq_length = 50
        X, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i - seq_length:i].flatten())
            y.append(scaled_data[i][0])
        X, y = np.array(X), np.array(y)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict next 50 days
        last_sequence = scaled_data[-seq_length:].flatten().tolist()
        predictions = []

        for _ in range(50):
            input_seq = np.array(last_sequence[-seq_length:]).reshape(1, -1)
            next_pred = model.predict(input_seq)[0]
            predictions.append(next_pred)
            last_sequence.append(next_pred)

        # Rescale predictions back to original prices
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Display results
        st.subheader("Predicted Stock Prices for Next 50 Days")
        st.bar_chart(predictions)

        # Plot actual vs predicted prices
        fig, ax = plt.subplots()
        ax.plot(data.index[-60:], data['Close'].values[-60:], label="Actual Prices", color="blue")
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=50)
        ax.plot(future_dates, predictions, label="Predicted Prices", color="red")

        ax.set_title(f"{ticker} - Actual vs Predicted Stock Prices (Random Forest)")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Stock Price")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
