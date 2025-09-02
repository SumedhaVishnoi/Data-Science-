import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Example: Sentiment classification (positive/negative)
model = Sequential()

# Word embedding layer (assume vocab size = 5000, each word = 32-dim vector)
model.add(Embedding(input_dim=5000, output_dim=32, input_length=100))

# RNN layer
model.add(SimpleRNN(64, activation='tanh'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
