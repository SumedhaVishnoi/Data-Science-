import torch

# Create a tensor
x = torch.tensor([1, 2, 3])
print(x)

# Random tensor
y = torch.rand(2, 3)   # 2 rows, 3 cols
print(y)

# Zeros and ones
z = torch.zeros(3, 3)
o = torch.ones(2, 2)

# Tensor from NumPy
import numpy as np
a = np.array([1, 2, 3])
t = torch.from_numpy(a)

if torch.cuda.is_available():
    x = x.to("cuda")   # moves tensor to GPU

# Requires gradient
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3*x + 4
y.backward()   # derivative wrt x
print(x.grad)  # dy/dx = 2x + 3 = 7

import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 1)  # input = 2 features, output = 1

    def forward(self, x):
        return torch.sigmoid(self.fc1(x))

# Model
model = SimpleNN()

# Loss & Optimizer
criterion = nn.BCELoss()               # Binary Cross Entropy
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR


import torch
import torch.nn as nn
import torch.optim as optim

# Sample data (X = inputs, y = labels)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Simple linear model: y = wx + b
model = nn.Linear(in_features=1, out_features=1)

# Loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # 1. Forward pass
    y_pred = model(X)

    # 2. Compute loss
    loss = criterion(y_pred, y)

    # 3. Backward pass
    optimizer.zero_grad()  # Reset gradients
    loss.backward()        # Backpropagation

    # 4. Update parameters
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

from torch.utils.data import DataLoader, TensorDataset

# Example dataset
X = torch.linspace(1, 100, 100).reshape(-1,1)
y = 3*X + 7 + torch.randn(X.size())*5   # y = 3x+7 + noise

# Create TensorDataset
dataset = TensorDataset(X, y)

# Create DataLoader (batch_size=10)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define model
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop with batches
for epoch in range(5):
    for batch_X, batch_y in loader:
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
