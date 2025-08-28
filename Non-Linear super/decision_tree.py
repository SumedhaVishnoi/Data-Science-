'''
 a tree like structure where data is split based on feature values 
 - can be used for both classification and regression 
 this is a non linearly seprable model 
 1 st  node - root node 
 2nd node - decision node 
 3rd node - leaf node ( finishing point )
 parent node - root node  , child node - sub nnode to a main node 
 to overcome we use pruning 
 pruning- reducing the size of the tree by removing parts that do not prove to provide power to the model 
 
  '''
  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

dataset = pd.read_csv()
dataset.head()
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder
sc = StandardScaler()
sc.fit(x)
x = pd.DataFrame(sc.transform(x))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)  

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

from sklearn.tree import plot_tree
plot_tree(model)
plt.show()

