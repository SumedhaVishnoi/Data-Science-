'''
used for both regression and classification 
non linaer model 
non-parametric algorithm 
also called lazy learner algorithm 
for prediction 
takes euclidean distance 
1. calculate distance between new data point and all training data points
2. sort the distances in ascending order    
3. select the top k nearest neighbors
4. for classification - take majority vote
5. for regression - take average of the k nearest neighbors
6. assign the class label or value to the new data point
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load example data (Iris dataset)
iris = load_iris()
X = iris.data  # feature matrix
y = iris.target  # target vector

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print("Accuracy:",knn.score(X_test,y_test))
