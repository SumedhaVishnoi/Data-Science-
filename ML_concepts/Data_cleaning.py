from altair import value
from anyio import __value
import pandas as pd 
dataset = pd.read_csv('data.csv')
dataset.head(10)
dataset.shape
dataset.isnull() # check for null values
dataset.isnull().sum() # total null values in each column
dataset.dropna(inplace=True) # drop rows with null values
dataset.shape # check shape after dropping null values  
dataset.duplicated() # check for duplicate values
dataset.duplicated().sum() # total duplicate values
dataset.drop_duplicates(inplace=True) # drop duplicate values
dataset.shape # check shape after dropping duplicate values
dataset.info() # get info about dataset
dataset.describe() # get statistical summary of dataset         
dataset.columns # get column names
dataset.dtypes # get data types of each column
dataset['column_name'].value_counts() # get value counts of a specific column
dataset['column_name'].unique() # get unique values of a specific column
dataset['column_name'].nunique() # get number of unique values in a specific column     
dataset['column_name'].fillna(value, inplace=True) # fill null values with a specific value
dataset['column_name'].replace(__value, __value, inplace=True) # replace old value with new value in a specific column
dataset['column_name'] = dataset['column_name'].astype('data_type') # change data type of a specific column
dataset['new_column'] = dataset['column1'] + dataset['column2'] # create a new column by combining two columns
dataset.to_csv('cleaned_data.csv', index=False) # save cleaned dataset to a new CSV file
dataset.corr() # get correlation between numerical columns

