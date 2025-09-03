'''
Simple linear regression 
relationship between one independent and one independent variable 
'''
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
# sample data 
X = np.array([1,2,3,4,5,6]).reshape(-1,1)   # Experience
y = np.array([1500, 1800, 2500, 2800, 3000, 3300])   # Salary

# train model 
model = LinearRegression()
model.fit(X, y)

# predictions 
y_pred = model.predict(X)

#equation 
print ("intercept:", model.intercept_)
print("slope:", model.coef_)

#visualization 
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('Experience (Years)')    
plt.ylabel('Salary ($)')
plt.title('Simple Linear Regression: Experience vs Salary')
plt.show()