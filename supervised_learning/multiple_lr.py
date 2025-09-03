'''
Multiple linear regeression 
Relationship between multiple independent variables (X1, X2, …, Xn) and Y.
'''
import pandas as pd
# Dataset
from sklearn.linear_model import LinearRegression


df = pd.DataFrame({
    'Size': [1000, 1500, 2000, 2500, 3000],
    'Rooms': [2, 3, 3, 4, 4],
    'Price': [200000, 250000, 300000, 350000, 400000]
})

X = df[['Size', 'Rooms']]
y = df['Price']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predictions
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Prediction Example
new_house = [[2200, 3]]
print("Predicted Price:", model.predict(new_house))
