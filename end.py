import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Set page title
st.set_page_config(page_title="Stock Market Predictor - LSTM vs CNN+LSTM vs Random Forest", layout="wide")
st.title("📈 Stock Price Prediction : LSTM vs CNN + LSTM vs Random Forest")


# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):").strip().upper()

if ticker:
    try:
        # Download stock data
        start_date = "2010-01-01"
        end_date = "2025-05-26"
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            st.error("No data found for this ticker. Please check the symbol.")
            st.stop()

        st.success(f"Data fetched successfully for {ticker}")
        st.write("Sample Data:", data.tail())

        close_prices = data['Close'].values.reshape(-1, 1)

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Load LSTM and CNN+LSTM models
        lstm_model = load_model("models/lstm_model.h5")
        cnn_lstm_model = load_model("models/cnn_lstm_model.h5")

        # Get sequence length from models
        seq_length_lstm = lstm_model.input_shape[1]
        seq_length_cnn_lstm = cnn_lstm_model.input_shape[1]

        min_seq_length = max(seq_length_lstm, seq_length_cnn_lstm, 50)  # include RF seq length (50)

        if len(scaled_data) < min_seq_length:
            st.error(f"Not enough historical data. Need at least {min_seq_length} days.")
            st.stop()

        # Prepare input sequences for LSTM models
        last_sequence_lstm = scaled_data[-seq_length_lstm:].reshape(1, seq_length_lstm, 1)
        last_sequence_cnn_lstm = scaled_data[-seq_length_cnn_lstm:].reshape(1, seq_length_cnn_lstm, 1)

        # Predict with LSTM
        lstm_predictions = []
        curr_seq_lstm = last_sequence_lstm.copy()
        for _ in range(50):
            pred = lstm_model.predict(curr_seq_lstm, verbose=0)
            lstm_predictions.append(pred[0][0])
            pred = pred.reshape(1, 1, 1)
            curr_seq_lstm = np.append(curr_seq_lstm[:, 1:, :], pred, axis=1)

        # Predict with CNN + LSTM
        cnn_lstm_predictions = []
        curr_seq_cnn_lstm = last_sequence_cnn_lstm.copy()
        for _ in range(50):
            pred = cnn_lstm_model.predict(curr_seq_cnn_lstm, verbose=0)
            cnn_lstm_predictions.append(pred[0][0])
            pred = pred.reshape(1, 1, 1)
            curr_seq_cnn_lstm = np.append(curr_seq_cnn_lstm[:, 1:, :], pred, axis=1)

        # Prepare training data for Random Forest
        seq_length_rf = 50
        X_rf, y_rf = [], []
        for i in range(seq_length_rf, len(scaled_data)):
            X_rf.append(scaled_data[i - seq_length_rf:i].flatten())
            y_rf.append(scaled_data[i][0])
        X_rf, y_rf = np.array(X_rf), np.array(y_rf)

        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_rf, y_rf)

        # Predict next 50 days with RF
        last_sequence_rf = scaled_data[-seq_length_rf:].flatten().tolist()
        rf_predictions = []
        for _ in range(50):
            input_seq = np.array(last_sequence_rf[-seq_length_rf:]).reshape(1, -1)
            next_pred = rf_model.predict(input_seq)[0]
            rf_predictions.append(next_pred)
            last_sequence_rf.append(next_pred)

        # Inverse transform all predictions
        lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
        cnn_lstm_predictions = scaler.inverse_transform(np.array(cnn_lstm_predictions).reshape(-1, 1))
        rf_predictions = scaler.inverse_transform(np.array(rf_predictions).reshape(-1, 1))

        # Future dates for plotting
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=50)

        # Display comparative chart
        st.markdown("### ⏳Predicted Next 50 Days Closing Prices Comparison")
        prediction_df = pd.DataFrame({
            'LSTM': lstm_predictions.flatten(),
            'CNN + LSTM': cnn_lstm_predictions.flatten(),
            'Random Forest': rf_predictions.flatten()
        }, index=future_dates)
        st.line_chart(prediction_df)

        # Plot Actual vs All Predictions
        st.subheader("📊 Actual vs LSTM vs CNN+LSTM vs Random Forest Predictions")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index[-100:], data['Close'][-100:], label="Actual Prices", color='blue')
        ax.plot(future_dates, lstm_predictions, label="LSTM Prediction", color='green', linestyle="--")
        ax.plot(future_dates, cnn_lstm_predictions, label="CNN + LSTM Prediction", color='red', linestyle="--")
        ax.plot(future_dates, rf_predictions, label="Random Forest Prediction", color='purple', linestyle="--")
        ax.axvline(data.index[-1], color='gray', linestyle='--', label='Prediction Start')
        ax.legend()
        ax.set_title(f"{ticker} - Actual vs Model Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
