'''Despite its name, Logistic Regression is a classification algorithm (not regression).
It predicts the probability of a data point belonging to a particular class.


should be linearly seprable 

types - binomial , multinomial , ordinal 

sigmoid function graph 


'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data
data = {'Hours': [1,2,3,4,5,6,7,8,9,10],
        'Pass':  [0,0,0,0,0,1,1,1,1,1]}

df = pd.DataFrame(data)

# Features & Target
X = df[['Hours']]
y = df['Pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))



# Binary classification with multiple inputs 
import numpy as np

# Dataset
data = {'Age': [22,25,47,52,46,56,55,60,30,40],
        'Salary': [25000,30000,50000,52000,49000,60000,58000,62000,40000,45000],
        'Bought_Insurance': [0,0,1,1,1,1,1,1,0,1]}

df = pd.DataFrame(data)

X = df[['Age','Salary']]
y = df['Bought_Insurance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

#  Binary classification with polynomail inputs 
from sklearn.preprocessing import PolynomialFeatures

# Polynomial transformation
poly = PolynomialFeatures(degree=2)  # quadratic features
X_poly = poly.fit_transform(df[['Hours']])  # using hours dataset again

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Multiclass classification
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target  # 0=Setosa, 1=Versicolor, 2=Virginica

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
