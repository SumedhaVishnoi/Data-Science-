''' fater 
handles missing values automatically
regularization 
sequntial 
each new tree fix the error of the previous trees 

'''

# Step 1: Install xgboost (if not installed)
# pip install xgboost

# Step 2: Import libraries
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 3: Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Initialize XGBoost
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
