'''
 values set before training a model 
 used to optimize model performance
 methods
 1. Grid Search 
 
 2. Cross-Validation'''
 
# 1) Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC     
from sklearn.datasets import load_iris
# Load dataset
data = load_iris()  
X = data.data
y = data.target
# Define model
model = SVC()
# Define hyperparameters and their possible values
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# Fit the model 
grid_search.fit(X, y)
# Best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# 2) Cross-Validation
from sklearn.model_selection import cross_val_score     
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
# Load dataset
data = load_iris()
X = data.data
y = data.target
# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", scores)