'''
SVM is a supervised ML algorithm used for classification (mainly) and regression.

It finds the best hyperplane that separates data into classes.

The margin between the classes and the hyperplane is maximized.

The support vectors are the data points closest to the hyperplane — they “support” the boundary.
'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]   # only first two features
y = (iris.target != 0) * 1   # convert to binary classification

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
model = SVC(kernel='linear')   # linear kernel
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
