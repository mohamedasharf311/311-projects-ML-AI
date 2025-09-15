import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Example data: A simple sine wave for demonstration
data = np.sin(np.linspace(0, 100, 1000))

# Prepare the data for LSTM
sequence_length = 50  # Number of previous time steps to predict the next
generator = TimeseriesGenerator(data, data, length=sequence_length, batch_size=32)

# Build the LSTM model
model = Sequential()

# Add LSTM layer
model.add(LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)))

# Add Dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Add Dense output layer
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error')

# Reshape data for LSTM input (samples, timesteps, features)
data = data.reshape((data.shape[0], 1, 1))

# Train the model
model.fit(generator, epochs=10, verbose=1)

# Make predictions
predictions = model.predict(generator)

# Example of using the model to predict the next time step
input_sequence = data[-sequence_length:]
input_sequence = input_sequence.reshape((1, sequence_length, 1))  # Reshape for prediction
next_value = model.predict(input_sequence)
print(f'Predicted next value: {next_value}')
