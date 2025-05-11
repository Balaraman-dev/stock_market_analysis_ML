# Import necessary libraries
import streamlit as st
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("models/lstm_model.h5")

# Streamlit app
st.title("Stock Market Prediction App")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., TSLA):")

if ticker:
    ticker = ticker.strip().upper()  # ✅ Ensure valid format

    try:
        # Fetch data
        data = yf.download(ticker, period="5y")
        if data.empty or 'Close' not in data.columns:
            st.error("No data found for the given stock ticker. Please check the ticker symbol.")
            st.stop()

        st.success(f"Data fetched successfully for {ticker}")
        st.write("Sample Data:", data.tail())  # ✅ Debug info

        close_prices = data['Close'].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Prepare input sequence
        seq_length = 60
        if len(scaled_data) < seq_length:
            st.error("Not enough historical data to make a prediction.")
            st.stop()

        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)

        # Predict next 50 days
        predictions = []
        for _ in range(50):
            pred = model.predict(last_sequence, verbose=0)  # suppress logs
            predictions.append(pred[0][0])
            pred = pred.reshape(1, 1, 1)
            last_sequence = np.append(last_sequence[:, 1:, :], pred, axis=1)

        # Rescale predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Display results
        st.subheader("Predicted Stock Prices for Next 50 Days")
        st.line_chart(predictions)

        # Plot actual vs predicted prices
        fig, ax = plt.subplots()
        ax.plot(data.index[-60:], data['Close'].values[-60:], label="Actual Prices", color="blue")
        ax.plot(range(len(data), len(data) + 50), predictions, label="Predicted Prices", color="red")
        ax.set_title(f"{ticker} - Actual vs Predicted Stock Prices")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Stock Price")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
