'''
Polynomial Regression using Linear Regression
Extends Linear Regression by adding polynomial terms (curves instead of straight line).
'''
from statistics import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.preprocessing import PolynomialFeatures

# Data
X = np.array([1,2,3,4,5,6]).reshape(-1,1)
y = np.array([1500, 1700, 2400, 3600, 5000, 8000])

# Transform to polynomial
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train Model
model = LinearRegression()
model.fit(X_poly, y)

# Predictions
y_pred = model.predict(X_poly)

# Visualization
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Polynomial Regression")
plt.show()
