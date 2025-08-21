import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 


 #central tendancy measures 
'''ar = np.array([4,5,6,2,1,8,5,6,4,7])
np.sum(ar)/len(ar)  # Mean
np.sort(ar)

#np.mean(ar)'''

# through dataset 
dataset = pd.read_csv("titanic.csv")
dataset.head(3)
dataset["age"].mean()
np.mean(dataset["age"]) #mean
sns.histplot(x="age",data=dataset)
plt.show 

np.median(dataset["age"]) #median 

dataset["age"].mode() #mode
dataset["age"].mode()[0]  # mode value


####################################################

# Measures of Variability 
dataset["age"].min()  # minimum value
dataset["age"].max()  # maximum value

dataset["age"].std()  # standard deviation
dataset["age"].var()  # variance    


##################################################

#measure of skewness- symmmetrical distribution has skewness of 0
dataset["age"].skew()  # skewness
#measure of kurtosis
dataset["age"].kurt()  # kurtosis   

