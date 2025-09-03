import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Example data
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7])

# Add constant for intercept
X_const = sm.add_constant(X)

# Build regression model
model = sm.OLS(y, X_const).fit()

# R-squared
print("R-squared:", model.rsquared)

# Adjusted R-squared
print("Adjusted R-squared:", model.rsquared_adj)
