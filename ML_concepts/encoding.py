'''#one hot encoding 
For every unique colum this makes a new binary column 

#Dummy variables 
similar to one hot but drops one category to avoid redundancy '''

#code 
from ast import Import
import pandas as pd 
dataset = pd.read_csv('data.csv')
dataset.head(3)
dataset.isnull().sum()
dataset["gender"].fillna(dataset["gender"].mode()[0], inplace=True) # fill null values with mode
en_data = dataset[["gender","married "]]
pd.get_dummies(en_data)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform
#fit transform - data ko analyse krke scikit process and then convert the data 

''' # Label Encoding 
converts categories into numbers by assigning an integer to each class 
'''
import pandas as pd 
# creating a data base 
df = pd.DataFrame({"name": ["wscube", "cow", "cat", "dog "]})
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
le.fit_transform
df["name"] = le.fit_transform(df["name"]) # fit and transform the data
print(df)


'''# Ordinal Encoding
used for categorical data with a defined order, like ratings or levels 
it assigns integers baased on the order or priority of categories '''

data = pd.DataFrame({"size": ["small", "medium", "large", "medium", "small"]})

from sklearn.preprocessing import ordinalEncoder 

oe = ordinalEncoder(categories=[["small", "medium", "large"]]) # defining the order
data["size"] = oe.fit_transform(data[["size"]])
print(data)
