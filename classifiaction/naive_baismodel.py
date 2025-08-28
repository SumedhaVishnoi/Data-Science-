'''A probabilistic classifier based on Bayes’ Theorem.

Assumes that all features are independent of each other given the class label → this is the “naive” assumption.

Types of Naive Bayes

Gaussian Naive Bayes → when features are continuous and assumed to follow normal distribution.

Multinomial Naive Bayes → works with count data (like word frequencies in text).

Bernoulli Naive Bayes → works with binary features (e.g., 0/1 word presence).'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
X, y = load_iris(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
