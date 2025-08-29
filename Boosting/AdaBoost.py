'''after training: 
    misclassified points- get higher weights 
    correctly classified points - get lower points 
    
ADVANTAGES:
    1. Can be used with any classifier
    2. Simple and easy to implement
    3. Can be used for both binary and multi-class classification problems
    4. Reduces bias and variance
    
DISADVANTAGES:
    1. Sensitive to noisy data and outliers
    2. Can overfit if the weak classifiers are too complex
    3. Requires careful tuning of parameters (e.g., number of estimators, learning rate)'''
    
# Step 1: Import libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2: Generate dataset
X, y = make_classification(n_samples=1000, n_features=10, 
                           n_informative=5, n_redundant=2,
                           random_state=42)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42)

# Step 4: Initialize weak learner (Decision Stump)
weak_learner = DecisionTreeClassifier(max_depth=1)

# Step 5: Create AdaBoost model
ada = AdaBoostClassifier(base_estimator=weak_learner, 
                         n_estimators=50, 
                         learning_rate=1.0,
                         random_state=42)

# Step 6: Train model
ada.fit(X_train, y_train)

# Step 7: Predictions
y_pred = ada.predict(X_test)

# Step 8: Accuracy
print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))

    
    