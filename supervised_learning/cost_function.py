''' A function that measures the error between actual values (y) and predicted values (ŷ).

The goal of training is to minimize the cost function so that predictions are as close as possible to the real data.
'''

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Actual and Predicted values
y_true = np.array([3, 5, 7, 9])
y_pred = np.array([2.5, 5.5, 6.8, 9.2])

# MSE
mse = mean_squared_error(y_true, y_pred)
print("MSE:", mse)

# RMSE
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# MAE
mae = mean_absolute_error(y_true, y_pred)
print("MAE:", mae)
