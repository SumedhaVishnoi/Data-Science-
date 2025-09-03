import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved!")

# Load model
with open("iris_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

print("✅ Model loaded!")
print("Prediction:", loaded_model.predict([X_test[0]]))
