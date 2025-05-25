import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# LSTM hybrid model + CNN
# Set page title
st.set_page_config(page_title="Stock Market Predictor - CNN+LSTM", layout="wide")
st.title("📈 Stock Price Prediction using CNN + LSTM")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL").strip().upper()

if ticker:
    try:
        # Download stock data
        start_date = "2010-01-01"
        end_date = "2025-04-01"
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("No data found for this ticker. Please check the symbol.")
            st.stop()

        st.success(f"Data fetched successfully for {ticker}")
        st.write("Sample Data:", data.tail())

        # Preprocessing
        close_prices = data['Close'].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Load trained CNN + LSTM model
        model = load_model("models/cnn_lstm_model.h5")

        # Prepare input sequence
        seq_length = model.input_shape[1]  # Get expected sequence length from model
        if len(scaled_data) < seq_length:
            st.error(f"Not enough historical data. Need at least {seq_length} days.")
            st.stop()

        last_sequence = scaled_data[-seq_length:]  # Shape: (seq_length, 1)
        last_sequence = last_sequence.reshape(1, seq_length, 1)  # Reshape for model

        # Predict next 50 days
        predictions = []
        for _ in range(50):
            pred = model.predict(last_sequence, verbose=0)
            predictions.append(pred[0][0])
            pred = pred.reshape(1, 1, 1)
            last_sequence = np.append(last_sequence[:, 1:, :], pred, axis=1)

        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Display prediction
        st.markdown("### 🔮 Predicted Next 50 Days Closing Prices")
        st.line_chart(predictions.flatten())

        # Plot Actual vs Predicted
        st.subheader("📊 Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index[-100:], data['Close'][-100:], label="Actual Prices", color='blue')
        future_dates = pd.date_range(start=data.index[-1], periods=51)[1:]  # Skip today's date
        ax.plot(future_dates, predictions, label="Predicted Prices", color='red', linestyle="--")
        ax.axvline(data.index[-1], color='gray', linestyle='--', label='Prediction Start')
        ax.legend()
        ax.set_title(f"{ticker} - Actual vs Predicted Prices")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Model: CNN + LSTM")