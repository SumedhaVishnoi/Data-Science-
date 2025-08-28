from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
log_clf = LogisticRegression(max_iter=1000)
knn_clf = KNeighborsClassifier()
dt_clf = DecisionTreeClassifier()

# Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(estimators=[
    ('lr', log_clf), ('knn', knn_clf), ('dt', dt_clf)
], voting='hard')

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
