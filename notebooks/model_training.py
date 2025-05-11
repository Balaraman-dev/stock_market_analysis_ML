import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load preprocessed data
X = np.load("../data/X.npy")
y = np.load("../data/y.npy")

# Debugging: Print the shapes of X and y to verify their dimensions
print("Shape of X before reshaping:", X.shape)
print("Shape of y:", y.shape)

# Ensure X has the correct shape (num_samples, sequence_length, num_features)
if len(X.shape) == 2:  # If X is 2D (e.g., (num_samples, sequence_length))
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape to 3D by adding a feature dimension

# Debugging: Print the final shape of X after reshaping
print("Shape of X after reshaping:", X.shape)

# Define LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),  # Use X.shape[2] for num_features
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, batch_size=32, epochs=50)

# Save the trained model
model.save("../models/lstm_model.h5")
print("Model trained and saved as ../models/lstm_model.h5.")