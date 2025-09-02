import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Build the model
model = Sequential([
    Dense(16, activation='relu', input_shape=(10,)),  # 10 features
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# 2. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Prepare dummy data for demonstration
import numpy as np
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
y_train = np.random.randint(0, 2, size=(100,))  # 100 binary labels

# 4. Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)



