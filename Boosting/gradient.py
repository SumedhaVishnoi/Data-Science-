'''focused on weights of misclassified points 
focuses on error 
builds model sequentially 

ADVANTAGES:
    1. pwerfull 
    2. works with both classification and regression 
    3. can capture complex patterns 
    
DISADVANTAGES:
    1. expensive and slower than random forest 
    2. can overfit when not overtuned 
    3. sensitive to hyperparameters like learning rate and number of trees '''
    
# Step 1: Import libraries
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 2: Generate dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=0.2, random_state=42)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Initialize Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Step 5: Train model
gb.fit(X_train, y_train)

# Step 6: Predictions
y_pred = gb.predict(X_test)

# Step 7: Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Gradient Boosting MSE:", mse)

























