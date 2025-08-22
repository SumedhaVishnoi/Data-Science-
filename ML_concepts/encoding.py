'''#one hot encoding 
For every unique colum this makes a new binary column 

#Dummy variables 
similar to one hot but drops one category to avoid redundancy '''

#code 
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