import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D

# Load preprocessed data
X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

# Debugging: Print the shapes of X and y
print("Shape of X before reshaping:", X.shape)
print("Shape of y:", y.shape)

# Ensure X has the correct shape: (num_samples, sequence_length, num_features)
if len(X.shape) == 2:
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Add feature dimension

print("Shape of X after reshaping:", X.shape)

# Define CNN + LSTM Hybrid Model
model = Sequential([
    # CNN Layers
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),

    # LSTM Layers
    LSTM(100, return_sequences=False),  # No return_sequences here
    Dropout(0.3),

    # Output Layer
    Dense(1)  # Predicting one future value (you can change to predict multiple steps)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Summary of the model
model.summary()

# Train the model
print("\nStarting training...\n")
history = model.fit(X, y, batch_size=32, epochs=50, validation_split=0.1)

# Save the trained model
model.save("../models/cnn_lstm_model.h5")
print("Model trained and saved as ../models/cnn_lstm_model.h5.")