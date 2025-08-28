a learning technique where multiple models are combined to build a stronger model with better accuracy and generalization 

TYPES :
1. bagging- reduce variance 
2. boosting- reduce bias 
3. stacking/ voting - combines different models predictions 

VOTING METHODS 
a. Max voting - for classification 
each model votes for a class and mojority is considered 

b. averaging- for regression 
predictions from all models are averaged 

weighted averaging 
some models are given higher weights if they perform better 


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

